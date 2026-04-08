# ================ base config ===================
version = 'mini'
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = './summary'

total_batch_size = 32
num_gpus = 8
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 100
checkpoint_epoch_interval = 20
occ_gt_range = [-40., -40., -1., 40., 40., 5.4]
occ_gt_shape = [200, 200, 16]
pc_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 7.4]
pad_occ_shape = [256, 256, 26]
occ_size = [0.4, 0.4, 0.4]
crop_size = [50, 50, 20]
eval_occ = True
gs_max_scale = 100

checkpoint_config = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)
log_config = dict(
    interval=51,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook", log_dir='./summary'),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)


# ================== model ========================
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


# 10 classes
occ_eval_class = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "barrier",
    "traffic_cone",
]

# # 8 classes
# occ_eval_class = [
#     "car",
#     "truck",
#     "construction_vehicle",
#     "bus",
#     "trailer",
#     "motorcycle",
#     "bicycle",
#     "pedestrian",
# ]

num_classes = len(class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4 # history + current

embed_dims = 256
gaussians_per_instance = 32
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True #False
use_checkpoint = False #True
with_gs_supervision = True #True

use_occ3d=True
use_cam_mask=True

scale_range = [0.0, 1.0] # may need to be modified
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'

task_config = dict(
    with_det=True,
)

model = dict(
    type="Guide",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="GuideHead",
        task_config=task_config,
        det_head=dict(
            type="Sparse4DHead",
            cls_threshold_to_reg=0.05,
            num_kpts_per_ins=gaussians_per_instance,
            with_gs_supervision=with_gs_supervision,
            use_checkpoint=use_checkpoint,
            crop_size=crop_size,
            voxel_size=0.4,
            decouple_attn=decouple_attn,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                per_num_gs=gaussians_per_instance,
                scale_range = scale_range,
                embed_dims=embed_dims,
                anchor="data/kmeans/kmeans_det_ego_900.npy",
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal else -1,
                confidence_decay=0.6,
                feat_grad=False,
                gs_feat_grad=False,
            ),
            anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            gs_anchor_encoder=dict(
				type="GSEncoder",
				embed_dims=[128, 96, 32] if decouple_attn else 256,
				mode="cat" if decouple_attn else "add",
				output_fc=not decouple_attn,
				in_loops=1,
				out_loops=4 if decouple_attn else 2,
			),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=(
                [
                    "deformable_ins",
                    "ffn_ins",
                    "norm_ins",
                    "refine_ins",
                ]
                * num_single_frame_decoder
                + [
                    "temp_gnn_ins",
                    "gnn_ins",
                    "norm_gs",
                    "sparse_conv_gs",
                    "norm_gs",
                    "deformable_gs",
                    "ffn_gs",
                    "norm_gs",
                    "refine_gs",
                ]
                * (num_decoder - num_single_frame_decoder)
            ),
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            )
            if temporal
            else None,
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer_ins=dict(type="LN", normalized_shape=embed_dims),
            ffn_ins=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            norm_layer_gs=dict(type="LN", normalized_shape=embed_dims),
            ffn_gs=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model_ins=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            refine_layer_ins=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_classes,
                refine_yaw=True,
                with_quality_estimation=with_quality_estimation,
            ),
            spconv_layer_gs=dict(
				type="SparseConv3D",
				in_channels=embed_dims, 
				embed_channels=embed_dims,
				pc_range=pc_range,
				grid_size=[0.2, 0.2, 0.2],
				phi_activation=phi_activation,
				xyz_coordinate=xyz_coordinate
			),
            deformable_model_gs=dict(
				type="DeformableFeatureAggregation",
				embed_dims=embed_dims,
				num_groups=num_groups,
				num_levels=num_levels,
				num_cams=6,
				attn_drop=0.15,
				use_deformable_func=use_deformable_func,
				use_camera_embed=True,
				residual_mode="cat",
				kps_generator=dict(
					type="SparseGaussian3DKeyPointsGenerator",
					phi_activation=phi_activation,
					xyz_coordinate=xyz_coordinate,
					num_learnable_pts=2,
					pc_range=pc_range,
					fix_scale=[
						[0, 0, 0],
						[0.45, 0, 0],
						[-0.45, 0, 0],
						[0, 0.45, 0],
						[0, -0.45, 0],
						[0, 0, 0.45],
						[0, 0, -0.45],
					],
				),
			),
			refine_layer_gs=dict(
				type="DualRefinementModule_GS",
				embed_dims=embed_dims,
				instance_output_dim=11,
				gs_group_output_dim=10,
				n=900,
				k=gaussians_per_instance,
				pc_range=pc_range,
				restrict_xyz=True,
				unit_xyz=[0.8, 0.8, 0.2],
				phi_activation=phi_activation,
				xyz_coordinate=xyz_coordinate,
				num_cls=num_classes,
				refine_yaw=True,
				with_quality_estimation=with_quality_estimation,
                gs_max_scale=gs_max_scale,
			),
            refine_layer=dict(
				type="RefinementModule",
				embed_dims=embed_dims,
				instance_output_dim=11,
				num_cls=num_classes,
				n=900,
				k=gaussians_per_instance,
				refine_yaw=True,
				with_quality_estimation=with_quality_estimation,
			),
            sampler=dict(
                type="SparseBox3DTarget",
                pc_range=pc_range,
                voxel_size=0.4,
                use_cam_mask=use_cam_mask,
                use_occ3d=use_occ3d,
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
                cls_allow_reverse=[class_names.index("barrier")],
            ),
            loss_vx=dict(
				type="VXFocalLoss",
				use_sigmoid=True,
				gamma=2.0,
				alpha=0.25,
				loss_weight=0.1,
			),
            decoder=dict(
                type="SparseBox3DDecoder",
                crop_size=crop_size,
                rel_range=pad_occ_shape,
                pc_range=pc_range,
                voxel_size=0.4,
                thr_vx=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.1],
                occ_score_threshold=0.2,
                occ_gt_range=occ_gt_range,
                occ_gt_shape=occ_gt_shape,
                occ_post_filter_range=[-45., -45., -5., 45., 45., 8],
                eval_occ=eval_occ,
                use_cam_mask=use_cam_mask,
            ),
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDataset"
data_root = "data/nuscenes/"
anno_root = "data/infos/" if version == 'trainval' else "data/infos/mini/"
occ_root = "data/occ3d_forground"
mask_root = "None"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="Lidar2egoTrans"),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_occ_instances',
            'gt_occ_semantics',
			'gt_camera_mask',
			'instance_inds',
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]
eval_pipeline = [
    dict(type="Lidar2egoTrans"),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='Collect', 
        keys=[
            'vectors',
            "gt_bboxes_3d",
            "gt_labels_3d",
        ],
        meta_keys=['token', 'timestamp']
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    occ_root=occ_root,
	mask_root=mask_root,
    classes=class_names,
    modality=input_modality,
    version="v1.0-trainval",
    use_occ3d=use_occ3d,
    occ_gt_range=occ_gt_range,
    pc_range=pc_range,
    pad_occ_shape=pad_occ_shape,
    occ_size=occ_size,
    occ_eval_class=occ_eval_class,
)
eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + 'nuscenes_infos_val.pkl',
    pipeline=eval_pipeline,
    test_mode=True,
)
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
)

# ================== eval ========================
eval_mode = dict(
    with_det=True,
    with_occ=eval_occ,
    with_tracking=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch*checkpoint_epoch_interval,
    eval_mode=eval_mode,
)