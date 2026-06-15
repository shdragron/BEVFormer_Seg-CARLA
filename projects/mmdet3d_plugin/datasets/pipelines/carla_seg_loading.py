"""Pipeline transforms for CARLA vehicle-occupancy seg (GaussianLSS-style image
load + GT mask formatting)."""
import numpy as np
import torch
from PIL import Image
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC

RS_W, RS_H, TOP_CROP = 480, 270, 46


@PIPELINES.register_module()
class LoadCarlaSegImages(object):
    """Load 6 images, resize 1600x900 -> 480x270, top-crop 46 -> 224x480
    (GaussianLSS transforms.py:143-169). Output mirrors
    LoadMultiViewImageFromFiles: results['img'] = list of 6 HWC float32."""

    def __call__(self, results):
        imgs = []
        for fn in results['img_filename']:
            im = Image.open(fn).convert('RGB').resize((RS_W, RS_H), Image.BILINEAR)
            im = im.crop((0, TOP_CROP, RS_W, RS_H))           # -> 224x480
            imgs.append(np.asarray(im).astype(np.float32))
        results['filename'] = results['img_filename']
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape
        results['ori_shape'] = imgs[0].shape
        results['pad_shape'] = imgs[0].shape
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.zeros(3, np.float32), std=np.ones(3, np.float32), to_rgb=False)
        return results


@PIPELINES.register_module()
class FormatCarlaSeg(object):
    """gt_seg / gt_valid (H,W) uint8 -> DC float tensor (1,H,W)."""

    def __call__(self, results):
        g = results['gt_seg'].astype(np.float32)[None]        # (1,H,W)
        results['gt_seg'] = DC(torch.from_numpy(g), stack=True)
        if 'gt_valid' in results:
            v = results['gt_valid'].astype(np.float32)[None]
            results['gt_valid'] = DC(torch.from_numpy(v), stack=True)
        return results
