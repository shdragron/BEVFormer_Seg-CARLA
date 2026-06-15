"""BEVFormer vehicle-occupancy BEV segmentation — R50 + 224x480 + BEV 200x200.

Fills the seg-task BEVFormer (Backward-projection) row. Matches the segmentation
benchmark protocol (CVT/LSS/GaussianLSS): 224x480 images, 200x200 BEV over
+-50m (0.5 m/cell), single vehicle binary occupancy, IoU@0.5. GaussianLSS GT
(vehicle = bit4 of bev png; see bevformer_seg/NOTES.md).
"""
_base_ = []
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]   # +-50m, 0.5m/cell @200
voxel_size = [0.5, 0.5, 8]
# ImageNet RGB norm (GaussianLSS uses none; R50-ImageNet needs it — NOTES.md)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                    to_rgb=True)
class_names = ['vehicle']

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 200

GAUSS = '/path/to/carla_geobev_root/carla_geobev_labels/gaussianlss'
IMG_ROOT = '/path/to/carla_geobev_root/carla_geobev'
SPLIT = '/path/to/carla_geobev_root/models/CARLA_GaussianLSS/GaussianLSS/data/splits/carla/train.txt'

model = dict(
    type='BEVFormerSeg',
    bev_h=bev_h_, bev_w=bev_w_, seg_inC=_dim_,
    dice_weight=1.0, bce_weight=1.0,
    use_grid_mask=False,
    video_test_mode=False,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(3,),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True, style='pytorch'),
    img_neck=dict(
        type='FPN', in_channels=[2048], out_channels=_dim_, start_level=0,
        add_extra_convs='on_output', num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEVFormerHead',
        bev_h=bev_h_, bev_w=bev_w_, num_query=900, num_classes=1,
        in_channels=_dim_, sync_cls_avg_factor=True, with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=False, use_shift=True, use_can_bus=False,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder', num_layers=3, pc_range=point_cloud_range,
                num_points_in_pillar=4, return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1),
                        dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                             deformable_attention=dict(
                                 type='MSDeformableAttention3D', embed_dims=_dim_,
                                 num_points=8, num_levels=_num_levels_),
                             embed_dims=_dim_)],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder', num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=_dim_,
                             num_heads=8, dropout=0.1),
                        dict(type='CustomMSDeformableAttention', embed_dims=_dim_,
                             num_levels=1)],
                    feedforward_channels=_ffn_dim_, ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range, max_num=300, voxel_size=voxel_size,
            num_classes=1),
        positional_encoding=dict(
            type='LearnedPositionalEncoding', num_feats=_pos_dim_,
            row_num_embed=bev_h_, col_num_embed=bev_w_),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                      loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)))

dataset_type = 'CarlaSegDataset'

train_pipeline = [
    dict(type='LoadCarlaSegImages'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='FormatCarlaSeg'),
    dict(type='CustomCollect3D', keys=['img', 'gt_seg', 'gt_valid'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape',
                    'scene_token', 'can_bus', 'prev_idx', 'next_idx', 'sample_idx')),
]
test_pipeline = [
    dict(type='LoadCarlaSegImages'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape',
                    'scene_token', 'can_bus', 'prev_idx', 'next_idx', 'sample_idx')),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(type=dataset_type, labels_dir=f'{GAUSS}/sedan', image_root=IMG_ROOT,
               split_file=SPLIT, pipeline=train_pipeline, bev_size=bev_h_,
               test_mode=False),
    val=dict(type=dataset_type, labels_dir=f'{GAUSS}/sedan_eval', image_root=IMG_ROOT,
             split_file=None, pipeline=test_pipeline, bev_size=bev_h_,
             test_mode=True, samples_per_gpu=1),
    test=dict(type=dataset_type, labels_dir=f'{GAUSS}/sedan_eval', image_root=IMG_ROOT,
              split_file=None, pipeline=test_pipeline, bev_size=bev_h_, test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(type='AdamW', lr=4e-4,
                 paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
                 weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500,
                 warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=2)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='BEVFormer_Seg',
                              name='bevformer_seg_r50_carla_sedan',
                              tags=['seg', 'vehicle', 'r50', 'bev200', 'sedan']))])
checkpoint_config = dict(interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/bevformer_seg_r50_carla_sedan'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
