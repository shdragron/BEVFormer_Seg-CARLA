# BEVFormer-Seg (CARLA) — Vehicle BEV Occupancy Segmentation

A BEVFormer-based model for **multi-camera vehicle BEV occupancy segmentation**
on CARLA, evaluated under the standard camera-only BEV-segmentation protocol
(224×480 images, 200×200 BEV over ±50 m, vehicle binary IoU).

It reuses BEVFormer's image backbone + BEV encoder (`PerceptionTransformer`,
spatial-cross / temporal-self attention) to produce a BEV embedding, then decodes
it to a 200×200 vehicle-occupancy map with a small ResNet-18 segmentation head,
supervised by BCE + Dice.

## Attribution / License

This repository is built on **BEVFormer**
(<https://github.com/fundamentalvision/BEVFormer>), released under the Apache
License 2.0. The original BEVFormer code (image backbone, BEV encoder, deformable
attention modules, training tools) is included under that license; see
[`LICENSE`](LICENSE). New code here — the segmentation detector/head, the CARLA
segmentation dataset, and its pipeline — is released under the same Apache-2.0
license.

## What's new on top of BEVFormer

| file | purpose |
|------|---------|
| `projects/mmdet3d_plugin/bevformer/detectors/bevformer_seg.py` | `BEVFormerSeg` detector + `SegHead` (ResNet-18 decoder), BCE + Dice |
| `projects/mmdet3d_plugin/datasets/carla_seg_dataset.py` | `CarlaSegDataset` — reads GaussianLSS-format scene JSONs directly |
| `projects/mmdet3d_plugin/datasets/pipelines/carla_seg_loading.py` | image loading (224×480) + GT formatting |
| `projects/configs/bevformer/bevformer_seg_r50_carla.py` | config: ResNet-50, 224×480, BEV 200×200, 24 epochs |

## Data format

GaussianLSS-style per-scene JSONs (`scene_XXXX.json`) with, per frame: 6 image
paths, intrinsics, ego→cam extrinsics, and a bit-packed BEV label PNG.

- **vehicle = bit 4** of the BEV PNG: `veh = (bev_png >> 4) & 1`.
- BEV grid: 200×200, 0.5 m/cell, ego-centered (`view = [[0,-2,100],[-2,0,100],[0,0,1]]`),
  ego origin at ground (cameras at z = +1.6 m).
- `lidar2img = K4 @ E`, `E` = ego→cam; intrinsics rescaled for the
  1600×900 → 480×270 resize + 46-px top crop → 224×480.

## Visibility protocol (IGNORE)

Following the standard camera-BEV-segmentation benchmark: cells with
`visibility < 2` are **ignored** in both the loss and the IoU (they are not
supervised or scored as background). Reported IoU is the **max over thresholds
{0.40, 0.45, 0.50}**, over cells with `visibility ≥ 2`.

## Setup

Same environment as BEVFormer (mmcv-full, mmdet, mmdet3d, mmsegmentation,
PyTorch). Follow the upstream BEVFormer install instructions, then edit the data
paths (`GAUSS`, `IMG_ROOT`, `SPLIT`) at the top of the config to point at your
CARLA / GaussianLSS-format data root.

## Train / evaluate

```bash
# train (2 GPUs)
bash tools/dist_train.sh projects/configs/bevformer/bevformer_seg_r50_carla.py 2

# evaluate a checkpoint
bash tools/dist_test.sh projects/configs/bevformer/bevformer_seg_r50_carla.py \
    work_dirs/bevformer_seg_r50_carla_sedan/latest.pth 2 --eval vehicle_IoU
```
