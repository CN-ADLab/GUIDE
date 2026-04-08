import os
import tqdm
import glob
import pickle
import argparse
import numpy as np
import torch
import multiprocessing
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box


parser = argparse.ArgumentParser()
parser.add_argument('--nusc-root', default='data/infos')
parser.add_argument('--occ3d-root', default='data/nuscenes/gts')
parser.add_argument('--output-dir', default='data/occ3d_forground')
parser.add_argument('--version', default='v1.0-trainval')
args = parser.parse_args()

token2path = {}
for gt_path in glob.glob(os.path.join(args.occ3d_root, '*/*/*.npz')):
    token = gt_path.split('/')[-2]
    token2path[token] = gt_path

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

det_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

foreground_class_names = [
    # 'traffic_cone',
    # 'truck',
    # 'car',
    # 'pedestrian',
    'movable_object.pushable_pullable',
    # 'construction_vehicle',
    # 'barrier',
    'movable_object.debris',
    # 'motorcycle',
    # 'bicycle',
    # 'bus',
    'static_object.bicycle_rack',
    # 'trailer',
    'human.pedestrian.stroller',
    'animal',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.wheelchair',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
]

def save_point_cloud_as_pcd(points, filename):
    # 保存为PCD文件
    with open(filename, "w") as file:
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z\n")
        file.write("SIZE 4 4 4\n")
        file.write("TYPE F F F\n")
        file.write("COUNT 1 1 1\n")
        file.write(f"WIDTH {points.shape[0]}\n")
        file.write("HEIGHT 1\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write(f"POINTS {points.shape[0]}\n")
        file.write("DATA ascii\n")

        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")


def convert_to_nusc_box(bboxes, lift_center=False, wlh_margin=0.0):
    results = []
    for q in range(bboxes.shape[0]):

        bbox = bboxes[q].copy()
        if lift_center:
            bbox[2] += bbox[5] * 0.5

        bbox_yaw = -bbox[6] - np.pi / 2
        orientation = Quaternion(axis=[0, 0, 1], radians=bbox_yaw).inverse

        box = Box(
            center=[bbox[0], bbox[1], bbox[2]],
            # 0.8 in pc range is roungly 2 voxels in occ grid
            # enlarge bbox to include voxels on the edge
            size=[bbox[3]+wlh_margin, bbox[4]+wlh_margin, bbox[5]+wlh_margin],
            orientation=orientation,
        )

        results.append(box)

    return results

def dense2sparse(occ_gt_dense):
    # 获取有效值的索引
    indices = np.argwhere(occ_gt_dense != -1)

    # 获取对应的值
    values = occ_gt_dense[indices[:, 0], indices[:, 1], indices[:, 2]]

    # 组合成 occ_gt_sparse
    occ_gt_sparse = np.hstack((indices, values[:, np.newaxis]))  # 添加新的一列作为值

    return occ_gt_sparse


def meshgrid3d(occ_size, pc_range):  # points in ego coord
    W, H, D = occ_size
    
    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz

def draw_box(ax, box):
    # 计算每个角的坐标
    w, l, h = box.wlh
    cx, cy, cz = box.center
    corners = np.array([
        [-l / 2, -w / 2, 0],
        [ l / 2, -w / 2, 0],
        [ l / 2,  w / 2, 0],
        [-l / 2,  w / 2, 0],
        [-l / 2, -w / 2, h],
        [ l / 2, -w / 2, h],
        [ l / 2,  w / 2, h],
        [-l / 2,  w / 2, h]
    ])

    # 应用旋转
    R_mat = box.rotation_matrix  # 获取旋转矩阵
    corners = corners @ R_mat.T  # 旋转

    # 平移到中心位置
    corners += box.center  

    # 各个面的边界
    edges = [
        [corners[0], corners[1], corners[2], corners[3], corners[0]],  # 底面
        [corners[4], corners[5], corners[6], corners[7], corners[4]],  # 顶面
        [corners[0], corners[4], corners[5], corners[1]],  # 连接底面和顶面的边
        [corners[1], corners[5], corners[6], corners[2]],  
        [corners[2], corners[6], corners[7], corners[3]],  
        [corners[3], corners[7], corners[4], corners[0]],  
    ]

    for edge in edges:
        # 画出每条边
        ax.plot3D(*zip(*edge), color='r', linewidth=0.5)


def process_add_instance_info(sample):
    point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
    occ_size = [200, 200, 16]
    num_classes = 18
    
    occ_gt_path = token2path[sample['token']]
    occ_labels = np.load(occ_gt_path)

    # save to original path
    data_scene = occ_gt_path.split(os.path.sep)[-3]
    data_path = os.path.join(data_scene, sample['lidar_token'], 'labels.npz')
    
    save_path = os.path.join(args.output_dir, data_path)
    
    save_dir = os.path.split(save_path)[0]
    if os.path.exists(save_dir):
        # print(f"文件已经存在，跳过保存: {save_dir}")
        return
    else:
        os.makedirs(save_dir)
    
    occ_gt = occ_labels['semantics']
    gt_boxes = sample['gt_boxes']
    gt_names = sample['gt_names']
    gt_ids = sample['instance_inds']
    
    bboxes = convert_to_nusc_box(gt_boxes)
    
    instance_gt = -np.ones(occ_gt.shape).astype(int)
    
    pts = meshgrid3d(occ_size, point_cloud_range).numpy()
    
    # filter out free voxels to accelerate
    valid_idx = np.where(occ_gt < 11)
    flatten_occ_gt = occ_gt[valid_idx]
    flatten_inst_gt = instance_gt[valid_idx]
    flatten_pts = pts[valid_idx]

    instance_boxes = []
    instance_class_ids = []
    gobj_ids = []
    gobj_sem = []
    instance_ids = []
    
    for i in range(len(gt_names)):
        if gt_names[i] in occ_class_names:
            occ_tag_id = occ_class_names.index(gt_names[i])
        else: 
            # print(data_path)
            occ_tag_id = 0
            gobj_tag_id = foreground_class_names.index(gt_names[i]) + num_classes
            
        # Move box to ego vehicle coord system
        bbox = bboxes[i]
        bbox.rotate(Quaternion(sample['lidar2ego_rotation']))
        bbox.translate(np.array(sample['lidar2ego_translation']))
        
        mask = points_in_box(bbox, flatten_pts.transpose(1, 0))

        # ignore voxels not belonging to this class
        mask[mask] = (flatten_occ_gt[mask] == occ_tag_id)
        # ignore voxels already occupied
        mask[mask] = (flatten_inst_gt[mask] == -1)

        
        # only instance with at least 1 voxel will be recorded
        if mask.sum() > 0:
            flatten_inst_gt[mask] = gt_ids[i] #instance_id

            instance_ids.append(gt_ids[i])
            if gt_names[i] not in occ_class_names:
                gobj_ids.append(gt_ids[i])
                gobj_sem.append(gobj_tag_id)
                flatten_occ_gt[mask] = gobj_tag_id
            
            # enlarge boxes to include voxels on the edge
            new_box = bbox.copy()
            new_box.wlh = new_box.wlh + 1.0
            
            instance_boxes.append(new_box)
            instance_class_ids.append(occ_tag_id)
    
    # post process unconvered non-occupied voxels
    uncover_idx = np.where(flatten_inst_gt == -1)
    uncover_pts = flatten_pts[uncover_idx]
    uncover_inst_gt = -np.ones_like(uncover_pts[..., 0]).astype(int)
    unconver_occ_gt = flatten_occ_gt[uncover_idx]
    
    # uncover_inst_dist records the dist between each voxel and its current nearest bbox's center
    uncover_inst_dist = np.ones_like(uncover_pts[..., 0]) * 1e8
    for i, box in enumerate(instance_boxes):
        # important, non-background inst id starts from 1
        # inst_id = i + 1
        class_id = instance_class_ids[i]
        mask = points_in_box(box, uncover_pts.transpose(1, 0))
        # mask voxels not belonging to this class
        mask[unconver_occ_gt != class_id] = False
        dist = np.sum((box.center - uncover_pts) ** 2, axis=-1)
        mask[dist >= uncover_inst_dist] = False

        if instance_ids[i] in gobj_ids:
            unconver_occ_gt[mask] = gobj_sem[gobj_ids.index(instance_ids[i])]
            flatten_occ_gt[uncover_idx] = unconver_occ_gt
        
        # important: only voxels inside the box (mask = True) and having no closer identical-class box need to update dist
        uncover_inst_gt[mask] = instance_ids[i]
        
    flatten_inst_gt[uncover_idx] = uncover_inst_gt
    
    instance_gt[valid_idx] = flatten_inst_gt
    occ_gt[valid_idx] = flatten_occ_gt
    
    # only semantic and mask information is needed to be reserved
    retain_keys = ['mask_lidar', 'mask_camera']   
    new_occ_labels = {k: occ_labels[k] for k in retain_keys}
    new_occ_labels['semantics'] = occ_gt
    new_occ_labels['instances'] = instance_gt
    np.savez_compressed(save_path, **new_occ_labels)


def add_instance_info(sample_infos):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    infos = sample_infos['infos']
    with tqdm.tqdm(total=len(infos)) as pbar:
            for info in infos:
                process_add_instance_info(info)
                pbar.update(1)


if __name__ == '__main__':
    if args.version == 'v1.0-trainval':
        sample_infos = pickle.load(open(os.path.join(args.nusc_root, 'nuscenes_infos_train.pkl'), 'rb'))
        add_instance_info(sample_infos)

        sample_infos = pickle.load(open(os.path.join(args.nusc_root, 'nuscenes_infos_val.pkl'), 'rb'))
        add_instance_info(sample_infos)

    elif args.version == 'v1.0-test':
        sample_infos = pickle.load(open(os.path.join(args.nusc_root, 'nuscenes_infos_test.pkl'), 'rb'))
        add_instance_info(sample_infos)

    else:
        raise ValueError