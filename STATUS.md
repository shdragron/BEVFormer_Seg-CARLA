# Development Status & Continuation Notes

State of the BEVFormer-Seg (CARLA) model, so work can resume from here.

## Current state

- Trained on CARLA (sedan vehicle class), **single-frame**, ResNet-50 backbone
  (**ImageNet pretrain only** — no depth/detection pretrain, `load_from=None`).
- In-distribution vehicle **IoU ≈ 0.43** (vis≥2 ignore, max over {0.40,0.45,0.50}).
- 24-epoch cosine schedule; val converges around **epoch 14–16**, then mild
  overfitting (train loss keeps dropping, val flat/slightly down). **Use the
  best-val checkpoint, not the last epoch.**

## Conventions (keep these to stay comparable with camera-BEV-seg baselines)

- **Visibility = IGNORE.** Cells with `visibility < 2` are masked out of **both
  the loss and the IoU** — they are NOT supervised or scored as background.
  (`gt_seg` = full vehicle occupancy; `gt_valid = (visibility ≥ 2)`; loss is
  computed only where `gt_valid==1`.)
- **IoU = max over thresholds {0.40, 0.45, 0.50}**, over `visibility ≥ 2` cells.
- **GT geometry**: vehicle = `bit 4` of the BEV label PNG; 200×200 BEV, 0.5 m/cell,
  ego origin at ground (cameras at z = +1.6 m); `lidar2img = K4 @ E` (E = ego→cam);
  intrinsics rescaled for 1600×900 → 480×270 + 46-px top crop → 224×480.

## Fixed gotchas (do not regress)

1. **Camera dimension** — `img` is `(B, 6, C, H, W)`; do NOT strip dim 1 (it is the
   6-camera axis, not a temporal queue). Only strip a genuine 6-dim queue input.
   (Stripping it = the model sees 1 camera → IoU collapses to ~0.05.)
2. **bs at eval** — `extract_img_feat` calls `img.squeeze_()` in place when bs==1,
   mutating `img` to `(6, C, H, W)`. Read `bs` from the **features**
   (`img_feats[0].size(0)`), not from the mutated `img`, in `_bev_embed`. Else the
   BEV grid gets bs=6 and the temporal-attention shapes mismatch at eval.

## Next steps

- Train the other vehicle classes: point `labels_dir` at the `suv` / `bus` data
  dirs (and `*_eval` for val); everything else in the config is unchanged.
- (Optional) robustness evaluation under camera-geometry perturbations.

## Run

See [README.md](README.md). Train: `bash tools/dist_train.sh
projects/configs/bevformer/bevformer_seg_r50_carla.py 2` (edit the data paths in
the config first). Evaluate the best checkpoint with `--eval vehicle_IoU`.
