"""CARLA vehicle-occupancy BEV segmentation dataset (GaussianLSS GT).

Reads the GaussianLSS-format scene jsons directly (no nuScenes DB) and yields the
dict BEVFormer's seg pipeline expects: 6x 224x480 images, ``lidar2img`` per cam,
and 200x200 binary masks in the BEVFormer grid layout.

Verified facts (see ``bevformer_seg/NOTES.md``, locked 2026-06-14):
  - vehicle = ``(bev_png >> 4) & 1``      (bit4; bit0/1 are road)
  - ``lidar2img = viewpad(K_rescaled) @ E``  (E = ego2cam; rescale = resize
    1600x900 -> 480x270 then top-crop 46, baked into the intrinsic)
  - GT layout to the BEVFormer grid: ``mask[::-1, ::-1].T``
  - visibility: IGNORE convention (GaussianLSS losses.py + metrics.py) — cells
    with ``visibility < min_visibility`` are masked out of both loss and IoU.
"""
import glob
import json
import os

import numpy as np
from PIL import Image
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
SRC_W, SRC_H = 1600.0, 900.0          # source image size
RS_W, RS_H = 480.0, 270.0             # after resize
TOP_CROP = 46                         # then top-crop -> 224x480
VEH_BIT = 4                           # vehicle bit in the packed BEV label PNG


@DATASETS.register_module()
class CarlaSegDataset(Dataset):
    CLASSES = ('vehicle',)

    def __init__(self, labels_dir, image_root, split_file, pipeline,
                 bev_size=200, test_mode=False, min_visibility=2, **kwargs):
        self.labels_dir = labels_dir
        self.image_root = image_root
        self.bev = bev_size
        self.test_mode = test_mode
        self.min_visibility = min_visibility

        if split_file:
            scenes = [s.strip() for s in open(split_file) if s.strip()]
        else:                          # all scene jsons present in labels_dir
            scenes = sorted(os.path.basename(p)[:-5] for p in
                            glob.glob(os.path.join(labels_dir, 'scene_*.json')))
        self.frames = []               # flat list of (scene, frame_idx, frame_dict)
        for s in scenes:
            jp = os.path.join(labels_dir, f'{s}.json')
            if not os.path.exists(jp):
                continue
            for fi, fr in enumerate(json.load(open(jp))):
                self.frames.append((s, fi, fr))
        self.flag = np.zeros(len(self.frames), dtype=np.uint8)   # mmdet sampler
        self.pipeline = Compose(pipeline)
        print(f'[CarlaSeg] {labels_dir.split("/")[-1]}: {len(scenes)} scenes -> '
              f'{len(self.frames)} frames')

    def __len__(self):
        return len(self.frames)

    # ----------------------------------------------------------- label loading
    def _read_png(self, scene, fname):
        return np.array(Image.open(os.path.join(self.labels_dir, scene, fname)))

    def _veh_mask(self, scene, bev_png):
        """Full vehicle occupancy (all visibilities) in the BEVFormer grid."""
        g = ((self._read_png(scene, bev_png) >> VEH_BIT) & 1).astype(np.uint8)
        return g[::-1, ::-1].T.copy()

    def _vis_mask(self, scene, fr):
        """Per-cell visibility in the BEVFormer grid (background = 255 sentinel)."""
        return self._read_png(scene, fr['visibility'])[::-1, ::-1].T.copy()

    @staticmethod
    def _rescale_K(K):
        """Intrinsic for the resize (1600x900->480x270) + top-crop 46."""
        K = np.array(K, dtype=np.float64).copy()
        K[0, 0] *= RS_W / SRC_W; K[0, 2] *= RS_W / SRC_W
        K[1, 1] *= RS_H / SRC_H; K[1, 2] *= RS_H / SRC_H
        K[1, 2] -= TOP_CROP
        return K

    # --------------------------------------------------------------- data dict
    def get_data_info(self, index):
        scene, fi, fr = self.frames[index]
        lidar2img = []
        for i in range(6):
            K4 = np.eye(4); K4[:3, :3] = self._rescale_K(fr['intrinsics'][i])
            E = np.array(fr['extrinsics'][i], dtype=np.float64)      # ego2cam
            lidar2img.append(K4 @ E)
        return dict(
            sample_idx=index,
            img_filename=[os.path.join(self.image_root, fr['images'][i]) for i in range(6)],
            lidar2img=lidar2img,
            can_bus=np.zeros(18, dtype=np.float64),
            scene_token=fr['token'],          # unique per frame -> prev_bev=None
            prev_idx='', next_idx='',
            gt_seg=self._veh_mask(scene, fr['bev']),                 # full vehicle GT
            gt_valid=(self._vis_mask(scene, fr) >= self.min_visibility).astype(np.uint8),
            img_shape_target=(int(RS_H - TOP_CROP), int(RS_W)),      # (224, 480)
        )

    def prepare(self, index):
        input_dict = self.get_data_info(index)
        input_dict['img_fields'] = []
        input_dict['img_prefix'] = None
        return self.pipeline(input_dict)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare(idx)
        while True:                       # skip frames the pipeline drops
            data = self.prepare(idx)
            if data is not None:
                return data
            idx = (idx + 1) % len(self)

    # ------------------------------------------------------------- evaluation
    def evaluate(self, results, logger=None, **kwargs):
        """IGNORE-convention IoU (GaussianLSS ``IoUMetric``): score ONLY cells
        with ``visibility >= min_visibility`` (low-vis cells excluded, not counted
        as FP), accumulate TP/FP/FN over the whole set at thresholds {0.4,0.45,0.5}
        and report the MAX-threshold IoU."""
        ths = [0.4, 0.45, 0.5]
        tp = [0] * 3; fp = [0] * 3; fn = [0] * 3
        for index, prob in enumerate(results):
            p = np.asarray(prob).reshape(self.bev, self.bev)
            scene, fi, fr = self.frames[index]
            g = self._veh_mask(scene, fr['bev']).astype(bool)
            valid = self._vis_mask(scene, fr) >= self.min_visibility
            pv, gv = p[valid], g[valid]
            for i, t in enumerate(ths):
                pb = pv >= t
                tp[i] += int((pb & gv).sum())
                fp[i] += int((pb & ~gv).sum())
                fn[i] += int((~pb & gv).sum())
        ious = [tp[i] / max(tp[i] + fp[i] + fn[i], 1) for i in range(3)]
        bi = int(np.argmax(ious))
        msg = (f'[CARLA-SEG] vehicle IoU (vis>={self.min_visibility}, '
               f'max@{ths[bi]:.2f}) = {ious[bi]:.4f}  '
               f'[@0.40={ious[0]:.4f} @0.45={ious[1]:.4f} @0.50={ious[2]:.4f}]')
        (logger.info if logger else print)(msg)
        return {'vehicle_IoU': ious[bi]}
