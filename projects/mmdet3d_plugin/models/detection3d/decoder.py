from typing import Optional
import numpy as np
import copy
import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F

from projects.mmdet3d_plugin.core.box3d import *
from local_aggregate_prob import LocalAggregator
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
import os

def decode_box(box):
    yaw = torch.atan2(box[..., SIN_YAW], box[..., COS_YAW])
    box = torch.cat(
        [
            box[..., [X, Y, Z]],
            box[..., [W, L, H]].exp(),
            yaw[..., None],
            box[..., VX:],
        ],
        dim=-1,
    )
    return box

@BBOX_CODERS.register_module()
class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
        crop_size: List = [30, 30, 10],
        rel_range: List = [200, 200, 16],
        pc_range: List = [-40., -40., -1., 40., 40., 5.4],
        voxel_size: int = 0.4,
        thr_vx: int = 0.15,
        multi_rate: int = 1,
        occ_score_threshold=None,
        occ_gt_range=[-40., -40., -1., 40., 40., 5.4],
        occ_gt_shape=[200, 200, 16],
        occ_post_filter_range=[-40., -40., -1., 40., 40., 5.4],
        eval_occ=False,
        use_cam_mask=True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

        self.rel_range = rel_range
        self.H, self.W, self.D = rel_range
        self.pc_range = pc_range
        self.crop_size = crop_size
        self.voxel_size = voxel_size 
        self.h, self.w, self.d = crop_size
        self.splatting = LocalAggregator(multi_rate, self.h, self.w, self.d, [-self.h * self.voxel_size / 2., -self.w * self.voxel_size / 2., -self.d * self.voxel_size / 2.], self.voxel_size)
        self.splatting.to('cuda')
        self.clamp_h = self.h * voxel_size / 2
        self.clamp_d = self.d * voxel_size / 2
        self.thr_vx = thr_vx
        self.occ_score_threshold = occ_score_threshold
        self.occ_gt_range = occ_gt_range
        self.occ_gt_shape = occ_gt_shape
        self.occ_post_filter_range = occ_post_filter_range
        self.eval_occ = eval_occ
        self.use_cam_mask = use_cam_mask

    def decode(
        self,
        cls_scores,
        box_preds,
        gs_preds,
        data,
        instance_id=None,
        quality=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        if self.eval_occ:
            gs_preds = gs_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.eval_occ:
                gs = gs_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
                if self.eval_occ:
                    gs = gs[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            #occ
            if self.eval_occ:
                #occ gt
                occ_pad_h = int((self.occ_gt_range[0] - self.pc_range[0]) / self.voxel_size)
                occ_pad_w = int((self.occ_gt_range[1] - self.pc_range[1]) / self.voxel_size)
                occ_pad_d = int((self.occ_gt_range[2] - self.pc_range[2]) / self.voxel_size)
                gt_occ_instances = data['gt_occ_instances'][i][occ_pad_h:occ_pad_h+self.occ_gt_shape[0], occ_pad_w:occ_pad_w+self.occ_gt_shape[1], occ_pad_d:occ_pad_d+self.occ_gt_shape[2]]
                occ_instance_id_set = gt_occ_instances[gt_occ_instances>-1]
                occ_instance_id_set = torch.unique(occ_instance_id_set)
                occ_gt_mask = torch.isin(data['instance_inds'][i], occ_instance_id_set)
                occ_gt_instance_inds = data['instance_inds'][i][occ_gt_mask]
                occ_gt_labels = data["gt_labels_3d"][i][occ_gt_mask]
                occ_gt_bboxes = data["gt_bboxes_3d"][i][occ_gt_mask]
                occ_gt_cam_mask = data["gt_camera_mask"][i][occ_pad_h:occ_pad_h+self.occ_gt_shape[0], occ_pad_w:occ_pad_w+self.occ_gt_shape[1], occ_pad_d:occ_pad_d+self.occ_gt_shape[2]]
                occ_gt_occ = []

                for occ_gt_instance_id in occ_gt_instance_inds:
                    if self.use_cam_mask:
                        occ_gt_instance = (gt_occ_instances==occ_gt_instance_id) & occ_gt_cam_mask
                    else:
                        occ_gt_instance = gt_occ_instances==occ_gt_instance_id

                    occ_gt_instance = occ_gt_instance.cpu().numpy()
                    occ_gt_occ.append(np.argwhere(occ_gt_instance))
                
                #occ pred
                occ_pred_category = copy.deepcopy(category_ids)
                occ_pred_box = copy.deepcopy(box)
                occ_pred_scores = copy.deepcopy(scores)
                occ_pred_gs = copy.deepcopy(gs)

                if self.occ_score_threshold is not None:
                    occ_pred_mask = occ_pred_scores > self.occ_score_threshold
                    occ_pred_category = occ_pred_category[occ_pred_mask]
                    occ_pred_box = occ_pred_box[occ_pred_mask]
                    occ_pred_scores = occ_pred_scores[occ_pred_mask]
                    occ_pred_gs = occ_pred_gs[occ_pred_mask]
                
                in_range_flags = ((occ_pred_box[:, 0] > self.occ_post_filter_range[0])
                        & (occ_pred_box[:, 1] > self.occ_post_filter_range[1])
                        & (occ_pred_box[:, 2] > self.occ_post_filter_range[2])
                        & (occ_pred_box[:, 0] < self.occ_post_filter_range[3])
                        & (occ_pred_box[:, 1] < self.occ_post_filter_range[4])
                        & (occ_pred_box[:, 2] < self.occ_post_filter_range[5]))
                
                occ_pred_category = occ_pred_category[in_range_flags]
                occ_pred_box = occ_pred_box[in_range_flags]
                occ_pred_scores = occ_pred_scores[in_range_flags]
                occ_pred_gs = occ_pred_gs[in_range_flags]

                occ_pred_occ = []

                for pred_cls in range(len(occ_pred_category)):
                    pred_occ_temp = self.cal_occ_single(occ_pred_box[pred_cls], occ_pred_gs[pred_cls], self.thr_vx[int(occ_pred_category[pred_cls])])
                    pred_occ_temp = pred_occ_temp[occ_pad_h+self.h//2:occ_pad_h+self.h//2+self.occ_gt_shape[0], occ_pad_w+self.w//2:occ_pad_w+self.w//2+self.occ_gt_shape[1], occ_pad_d+self.d//2:occ_pad_d+self.d//2+self.occ_gt_shape[2]]

                    #post-processing
                    box_occ = self.bbox_to_voxels(occ_pred_box[pred_cls][:7].cpu().numpy(), self.voxel_size, (200, 200, 16), np.array([-40, -40, -1]))
                    box_occ = torch.from_numpy(box_occ).to(pred_occ_temp.device)
                    pred_occ_temp = pred_occ_temp & box_occ
                    if occ_pred_category[pred_cls] in [1, 2, 3, 4]:
                        temp_box = occ_pred_box[pred_cls][:7].cpu().numpy()
                        temp_box[3:6] = temp_box[3:6]*0.8
                        box_occ = self.bbox_to_voxels(temp_box, self.voxel_size, (200, 200, 16), np.array([-40, -40, -1]))
                        box_occ = torch.from_numpy(box_occ).to(pred_occ_temp.device)
                        pred_occ_temp = pred_occ_temp | box_occ

                    if self.use_cam_mask:
                        pred_occ_temp = pred_occ_temp & occ_gt_cam_mask
                    pred_occ_temp = pred_occ_temp.cpu().numpy()
                    occ_pred_occ.append(np.argwhere(pred_occ_temp))
                
                output.append(
                    {
                        "boxes_3d": box.cpu(),
                        "scores_3d": scores.cpu(),
                        "labels_3d": category_ids.cpu(),
                        "occ_gt_labels": occ_gt_labels.cpu().numpy(),
                        "occ_gt_occ": occ_gt_occ,
                        "occ_pred_scores": occ_pred_scores.cpu().numpy(),
                        "occ_pred_category": occ_pred_category.cpu().numpy(),
                        "occ_pred_occ": occ_pred_occ,
                    }
                )
            else:
                output.append(
                    {
                        "boxes_3d": box.cpu(),
                        "scores_3d": scores.cpu(),
                        "labels_3d": category_ids.cpu(),
                    }
                )

            if quality is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids
        return output

    @staticmethod
    def bbox_to_voxels(bbox, voxel_size, grid_size, grid_lower_bound):

        x, y, z, lh, lw, ld, yaw = bbox
        
        c, s = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        corners = np.array([
            [-lh / 2, -lw / 2],
            [ lh / 2, -lw / 2],
            [ lh / 2,  lw / 2],
            [-lh / 2,  lw / 2]
        ])

        transformed_corners = np.dot(corners, rotation_matrix)
        
        global_corners = transformed_corners + np.array([[x, y]])

        min_x_idx = int(np.floor((global_corners[:, 0].min() - grid_lower_bound[0]) / voxel_size))
        max_x_idx = int(np.ceil((global_corners[:, 0].max() - grid_lower_bound[0]) / voxel_size))

        min_y_idx = int(np.floor((global_corners[:, 1].min() - grid_lower_bound[1]) / voxel_size))
        max_y_idx = int(np.ceil((global_corners[:, 1].max() - grid_lower_bound[1]) / voxel_size))

        min_z_idx = int(np.floor((z - ld / 2 - grid_lower_bound[2]) / voxel_size))
        max_z_idx = int(np.ceil((z + ld / 2 - grid_lower_bound[2]) / voxel_size))

        voxel_grid = np.zeros(grid_size, dtype=bool)

        for i in range(max(0, min_x_idx), min(grid_size[0], max_x_idx)):
            for j in range(max(0, min_y_idx), min(grid_size[1], max_y_idx)):
                for k in range(max(0, min_z_idx), min(grid_size[2], max_z_idx)):
                    voxel_grid[i, j, k] = True

        return voxel_grid

    def cal_occ_single(self, pred_box, gaussians, thr_occ):
        centers = self.gen_centers(self.h, self.w, self.d, self.voxel_size).to(gaussians.device).flatten(end_dim=-2)
        centers[:,0] = centers[:,0]-self.h * self.voxel_size / 2
        centers[:,1] = centers[:,1]-self.w * self.voxel_size / 2
        centers[:,2] = centers[:,2]-self.d * self.voxel_size / 2

        gaussians[..., :2] = torch.clamp(gaussians[..., :2], -self.clamp_h+1e-2, self.clamp_h-1e-2)
        gaussians[..., 2:3] = torch.clamp(gaussians[..., 2:3], -self.clamp_d+1e-2, self.clamp_d-1e-2)
        k = gaussians.shape[0]
        scales = gaussians[..., 3:6].clone()
        rotations = gaussians[..., 6:10].clone()
        S = torch.zeros(k, 3, 3, dtype=gaussians.dtype, device=gaussians.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = self.get_rotation_matrix(rotations) # n, k, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = torch.inverse(Cov) # n, k, 3, 3
        vx = self.splatting(centers, gaussians[:, :3], gaussians[:, 3:6], CovInv).reshape(self.h, self.w, self.d).to(dtype=torch.float)

        idx = ((torch.div((pred_box[:3] - gaussians.new_tensor(self.pc_range[:3])), self.voxel_size))).to(int)

        idx[:2] = torch.clamp(idx[:2], 0, self.rel_range[0]-1)
        idx[2:] = torch.clamp(idx[2:], 0, self.rel_range[2]-1)

        result = centers.new_full([self.H + self.h, self.W + self.w, self.D + self.d], False, dtype=torch.bool)

        result[
            idx[0]: idx[0] + vx.shape[0],
            idx[1]: idx[1] + vx.shape[1],
            idx[2]: idx[2] + vx.shape[2]
        ] = torch.where(vx>=thr_occ, True, False)

        return result

    def gs2vx(self, gaussians):
        centers = self.gen_centers(self.h, self.w, self.d, self.voxel_size).to(gaussians.device).flatten(end_dim=-2)
        centers[:,0] = centers[:,0]-self.h * self.voxel_size / 2
        centers[:,1] = centers[:,1]-self.w * self.voxel_size / 2
        centers[:,2] = centers[:,2]-self.d * self.voxel_size / 2

        gaussians[..., :2] = torch.clamp(gaussians[..., :2], -self.clamp_h+1e-2, self.clamp_h-1e-2)
        gaussians[..., 2:3] = torch.clamp(gaussians[..., 2:3], -self.clamp_d+1e-2, self.clamp_d-1e-2)
        n, k = gaussians.shape[:2]
        scales = gaussians[..., 3:6].clone()
        rotations = gaussians[..., 6:10].clone()
        S = torch.zeros(n, k, 3, 3, dtype=gaussians.dtype, device=gaussians.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = self.get_rotation_matrix(rotations) # n, k, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = torch.inverse(Cov) # n, k, 3, 3

        ins_voxel = (
            gaussians.new_zeros([n, self.h, self.w, self.d], dtype=torch.float)
            )

        for i in range(gaussians.shape[0]):
            gs = gaussians[i]
            covinv = CovInv[i]
            vx = self.splatting(centers, gs[:, :3], gs[:, 10:], gs[:, 3:6], covinv).squeeze(1).reshape(self.h, self.w, self.d).to(dtype=torch.float)
            ins_voxel[i] = vx

        return ins_voxel

    def gen_centers(self, H, W, D, voxel_size):

        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H-1, H),
            torch.linspace(0, W-1, W),
            torch.linspace(0, D-1, D)
        )

        centers_z = grid_z * voxel_size
        centers_y = grid_y * voxel_size
        centers_x = grid_x * voxel_size

        centers = torch.stack((centers_z, centers_y, centers_x), dim=3)

        return centers

    def get_rotation_matrix(self, tensor):
        assert tensor.shape[-1] == 4

        tensor = F.normalize(tensor, dim=-1)
        mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
        mat1[..., 0, 0] = tensor[..., 0]
        mat1[..., 0, 1] = - tensor[..., 1]
        mat1[..., 0, 2] = - tensor[..., 2]
        mat1[..., 0, 3] = - tensor[..., 3]
        
        mat1[..., 1, 0] = tensor[..., 1]
        mat1[..., 1, 1] = tensor[..., 0]
        mat1[..., 1, 2] = - tensor[..., 3]
        mat1[..., 1, 3] = tensor[..., 2]

        mat1[..., 2, 0] = tensor[..., 2]
        mat1[..., 2, 1] = tensor[..., 3]
        mat1[..., 2, 2] = tensor[..., 0]
        mat1[..., 2, 3] = - tensor[..., 1]

        mat1[..., 3, 0] = tensor[..., 3]
        mat1[..., 3, 1] = - tensor[..., 2]
        mat1[..., 3, 2] = tensor[..., 1]
        mat1[..., 3, 3] = tensor[..., 0]

        mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
        mat2[..., 0, 0] = tensor[..., 0]
        mat2[..., 0, 1] = - tensor[..., 1]
        mat2[..., 0, 2] = - tensor[..., 2]
        mat2[..., 0, 3] = - tensor[..., 3]
        
        mat2[..., 1, 0] = tensor[..., 1]
        mat2[..., 1, 1] = tensor[..., 0]
        mat2[..., 1, 2] = tensor[..., 3]
        mat2[..., 1, 3] = - tensor[..., 2]

        mat2[..., 2, 0] = tensor[..., 2]
        mat2[..., 2, 1] = - tensor[..., 3]
        mat2[..., 2, 2] = tensor[..., 0]
        mat2[..., 2, 3] = tensor[..., 1]

        mat2[..., 3, 0] = tensor[..., 3]
        mat2[..., 3, 1] = tensor[..., 2]
        mat2[..., 3, 2] = - tensor[..., 1]
        mat2[..., 3, 3] = tensor[..., 0]

        mat2 = torch.conj(mat2).transpose(-1, -2)
        
        mat = torch.matmul(mat1, mat2)
        return mat[..., 1:, 1:]