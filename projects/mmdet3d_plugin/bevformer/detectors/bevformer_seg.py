"""BEVFormer vehicle-occupancy BEV segmentation detector.

Reuses the BEVFormer image backbone + BEV encoder (PerceptionTransformer) to
produce a (bev_h*bev_w, bs, C) BEV embedding, then decodes it to a 200x200
vehicle-occupancy logit with a small SegEncode head, supervised by BCE + Dice.
Single-frame (queue_length=1): prev_bev is always None (unique scene_token).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from .bevformer import BEVFormer


class _Up(nn.Module):
    def __init__(self, cin, cout, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], 1))


class SegHead(nn.Module):
    """SegEncode (BEVFormer-seg repo) — spatial-size preserving resnet18 decoder."""
    def __init__(self, inC, outC=1):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, 7, stride=2, padding=3, bias=False)
        self.bn1, self.relu = trunk.bn1, trunk.relu
        self.layer1, self.layer2, self.layer3 = trunk.layer1, trunk.layer2, trunk.layer3
        self.up1 = _Up(64 + 256, 256, scale=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, outC, 1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x); x = self.layer2(x1); x2 = self.layer3(x)
        return self.up2(self.up1(x2, x1))             # (bs,outC,H,W)


@DETECTORS.register_module()
class BEVFormerSeg(BEVFormer):
    def __init__(self, *args, bev_h=200, bev_w=200, seg_inC=256,
                 dice_weight=1.0, bce_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bev_h, self.bev_w = bev_h, bev_w
        self.seg_head = SegHead(seg_inC, 1)
        self.dice_w, self.bce_w = dice_weight, bce_weight

    def _bev_embed(self, img, img_metas):
        """Run backbone + BEV encoder -> (bs, C, bev_h, bev_w)."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # NB: extract_img_feat does img.squeeze_() in-place when bs==1, so
        # img.size(0) would read 6 (the camera count) at eval. Take bs from the
        # features (B, N, C, H, W), which is what get_bev_features uses too.
        bs = img_feats[0].size(0)
        dtype = img_feats[0].dtype
        bev_queries = self.pts_bbox_head.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.pts_bbox_head.positional_encoding(bev_mask).to(dtype)
        bev = self.pts_bbox_head.transformer.get_bev_features(
            img_feats, bev_queries, self.bev_h, self.bev_w,
            grid_length=(self.pts_bbox_head.real_h / self.bev_h,
                         self.pts_bbox_head.real_w / self.bev_w),
            bev_pos=bev_pos, img_metas=img_metas, prev_bev=None)
        # bev_embed is (bev_h*bev_w, bs, C) (HW-first, verified vs seg repo);
        # be robust to (bs, HW, C) too. Unflatten row-major (H,W), W contiguous.
        C = bev.shape[-1]
        if bev.shape[0] == bs and bev.shape[1] == self.bev_h * self.bev_w:
            bev = bev.permute(1, 0, 2)            # -> (HW, bs, C)
        return bev.permute(1, 2, 0).contiguous().view(
            bs, C, self.bev_h, self.bev_w)

    def forward_train(self, img_metas=None, img=None, gt_seg=None,
                      gt_valid=None, **kwargs):
        if isinstance(img_metas[0], list):           # unwrap queue/list nesting
            img_metas = [m[-1] for m in img_metas]
        if img.dim() == 6:                           # genuine queue dim -> last frame
            img = img[:, -1]
        # img is now (bs, N_cam, C, H, W) — what extract_feat expects. Do NOT
        # strip dim 1 here: it is the 6-camera axis, not a temporal queue.
        bev = self._bev_embed(img, img_metas)
        logit = self.seg_head(bev)                    # (bs,1,200,200)
        gt = gt_seg.float()
        if gt.dim() == 3:
            gt = gt[:, None]
        # IGNORE convention (matches the seg benchmark, GaussianLSS losses.py):
        # supervise ONLY cells with visibility>=min (gt_valid==1). Low-vis cells
        # are masked out — never pushed to background.
        if gt_valid is not None:
            v = gt_valid.float()
            if v.dim() == 3:
                v = v[:, None]
        else:
            v = torch.ones_like(gt)
        denom = v.sum().clamp(min=1.0)
        bce_map = F.binary_cross_entropy_with_logits(logit, gt, reduction='none')
        bce = (bce_map * v).sum() / denom
        p = torch.sigmoid(logit) * v
        gtv = gt * v
        dice = 1 - (2 * (p * gtv).sum() + 1) / (p.sum() + gtv.sum() + 1)
        return dict(loss_bce=self.bce_w * bce, loss_dice=self.dice_w * dice)

    def forward_test(self, img_metas=None, img=None, **kwargs):
        if isinstance(img_metas[0], list):
            img_metas = [m[-1] for m in img_metas]
        if isinstance(img, list):
            img = img[0]
        if img.dim() == 6:                           # genuine queue dim -> last frame
            img = img[:, -1]
        # keep (bs, N_cam, C, H, W): dim 1 is the 6-camera axis, not a queue.
        bev = self._bev_embed(img, img_metas)
        prob = torch.sigmoid(self.seg_head(bev))[:, 0]     # (bs,200,200)
        return prob.detach().cpu().numpy()
