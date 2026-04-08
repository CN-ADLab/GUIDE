from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn.bricks.registry import (
	ATTENTION,
	PLUGIN_LAYERS,
	POSITIONAL_ENCODING,
	FEEDFORWARD_NETWORK,
	NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from ..blocks import DeformableFeatureAggregation as DFG

from local_aggregate_prob import LocalAggregator

__all__ = ["Sparse4DHead"]


@HEADS.register_module()
class Sparse4DHead(BaseModule):
	def __init__(
		self,
		instance_bank: dict,
		anchor_encoder: dict,
		gs_anchor_encoder: dict,
		graph_model: dict,
		norm_layer_ins: dict,
		ffn_ins: dict,
		norm_layer_gs: dict,
		ffn_gs: dict,
		deformable_model_ins: dict,
		refine_layer_ins: dict,
		deformable_model_gs: dict,
		spconv_layer_gs: dict,
		refine_layer_gs: dict,
		refine_layer: dict,
		num_decoder: int = 6,
		num_single_frame_decoder: int = -1,
		num_kpts_per_ins: int = 64,
		temp_graph_model: dict = None,
		loss_cls: dict = None,
		loss_reg: dict = None,
		loss_vx: dict = None,
		decoder: dict = None,
		sampler: dict = None,
		gt_cls_key: str = "gt_labels_3d",
		gt_reg_key: str = "gt_bboxes_3d",
		gt_id_key: str = "instance_id",
		gt_ins_occ_key: str = "gt_occ_instances",
		gt_camera_mask: str = "gt_camera_mask",
		gt_ins_id: str = "instance_inds",
		with_instance_id: bool = True,
		task_prefix: str = 'det',
		reg_weights: List = None,
		operation_order: Optional[List[str]] = None,
		cls_threshold_to_reg: float = -1,
		decouple_attn: bool = True,
		init_cfg: dict = None,
		with_gs_supervision: bool = False,
		crop_size: List = [60, 60, 24],
		voxel_size: int = 0.2,
		use_checkpoint: bool = False,
		multi_rate = 1,
		**kwargs,
	):
		super(Sparse4DHead, self).__init__(init_cfg)
		self.num_decoder = num_decoder
		self.num_kpts_per_ins = num_kpts_per_ins
		self.num_single_frame_decoder = num_single_frame_decoder
		self.gt_cls_key = gt_cls_key
		self.gt_reg_key = gt_reg_key
		self.gt_id_key = gt_id_key
		self.gt_occ_key = gt_ins_occ_key
		self.gt_camera_mask = gt_camera_mask
		self.gt_ins_id = gt_ins_id
		self.with_instance_id = with_instance_id
		self.task_prefix = task_prefix
		self.cls_threshold_to_reg = cls_threshold_to_reg
		self.decouple_attn = decouple_attn
		self.use_checkpoint = use_checkpoint

		if reg_weights is None:
			self.reg_weights = [1.0] * 10
		else:
			self.reg_weights = reg_weights

		if operation_order is None:
			operation_order = [
				"temp_gnn",
				"gnn",
				"norm",
				"deformable",
				"norm",
				"ffn",
				"norm",
				"refine",
			] * num_decoder
			operation_order = operation_order[3:]
		self.operation_order = operation_order

		# =========== build modules ===========
		def build(cfg, registry):
			if cfg is None:
				return None
			return build_from_cfg(cfg, registry)

		self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
		self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
		self.gs_anchor_encoder = build(gs_anchor_encoder, POSITIONAL_ENCODING)
		self.sampler = build(sampler, BBOX_SAMPLERS)
		self.decoder = build(decoder, BBOX_CODERS)
		self.loss_cls = build(loss_cls, LOSSES)
		self.loss_reg = build(loss_reg, LOSSES)
		self.op_config_map = {
			"temp_gnn_ins": [temp_graph_model, ATTENTION],
			"gnn_ins": [graph_model, ATTENTION],
			"norm_ins": [norm_layer_ins, NORM_LAYERS],
			"ffn_ins": [ffn_ins, FEEDFORWARD_NETWORK],
			"deformable_ins": [deformable_model_ins, ATTENTION],
			"refine_ins": [refine_layer_ins, PLUGIN_LAYERS],
			"deformable_gs": [deformable_model_gs, ATTENTION],
			"sparse_conv_gs": [spconv_layer_gs, ATTENTION],
			"norm_gs": [norm_layer_gs, NORM_LAYERS],
			"ffn_gs": [ffn_gs, FEEDFORWARD_NETWORK],
			"refine_gs": [refine_layer_gs, PLUGIN_LAYERS],
			"refine": [refine_layer, PLUGIN_LAYERS],
		}
		self.layers = nn.ModuleList(
			[
				build(*self.op_config_map.get(op, [None, None]))
				for op in self.operation_order
			]
		)
		self.embed_dims = self.instance_bank.embed_dims
		if self.decouple_attn:
			self.fc_before = nn.Linear(
				self.embed_dims, self.embed_dims * 2, bias=False
			)
			self.fc_after = nn.Linear(
				self.embed_dims * 2, self.embed_dims, bias=False
			)
		else:
			self.fc_before = nn.Identity()
			self.fc_after = nn.Identity()

		self.with_gs_supervision = with_gs_supervision
		self.crop_size = crop_size
		self.voxel_size = voxel_size 
		self.h, self.w, self.d = crop_size
		self.clamp_h = self.h * voxel_size / 2
		self.clamp_d = self.d * voxel_size / 2

		if self.with_gs_supervision:
			self.splatting = LocalAggregator(multi_rate, self.h, self.w, self.d, [-self.h * self.voxel_size / 2., -self.w * self.voxel_size / 2., -self.d * self.voxel_size / 2.], self.voxel_size)
			self.loss_vx = build(loss_vx, LOSSES)

	def init_weights(self):
		for i, op in enumerate(self.operation_order):
			if self.layers[i] is None:
				continue
			elif op != "refine" and op != "refine_ins" and op != "refine_gs":
				for p in self.layers[i].parameters():
					if p.dim() > 1:
						nn.init.xavier_uniform_(p)
		for m in self.modules():
			if hasattr(m, "init_weight"):
				m.init_weight()

	def graph_model(
		self,
		index,
		query,
		key=None,
		value=None,
		query_pos=None,
		key_pos=None,
		**kwargs,
	):
		if self.decouple_attn:
			query = torch.cat([query, query_pos], dim=-1)
			if key is not None:
				key = torch.cat([key, key_pos], dim=-1)
			query_pos, key_pos = None, None
		if value is not None:
			value = self.fc_before(value)
		return self.fc_after(
			self.layers[index](
				query,
				key,
				value,
				query_pos=query_pos,
				key_pos=key_pos,
				**kwargs,
			)
		)

	def forward(
		self,
		feature_maps: Union[torch.Tensor, List],
		metas: dict,
	):
		if isinstance(feature_maps, torch.Tensor):
			feature_maps = [feature_maps]
		batch_size = feature_maps[0].shape[0]

		# ========= get instance info ============
		(
			instance_feature,
			anchor,
			init_gs_feature,
			gs_anchor,
			temp_instance_feature,
			temp_anchor,
			time_interval,
		) = self.instance_bank.get(batch_size, metas)

		num_anchor = anchor.shape[1]

		attn_mask = None

		anchor_embed = self.anchor_encoder(anchor)
		if temp_anchor is not None:
			temp_anchor_embed = self.anchor_encoder(temp_anchor)
		else:
			temp_anchor_embed = None

		# =================== forward the layers ====================
		prediction = []
		classification = []
		quality = []
		prediction_gs = []
		for i, op in enumerate(self.operation_order):
			if self.layers[i] is None:
				continue
			elif op == "norm_ins" or op == "ffn_ins":
				if self.use_checkpoint and self.training:
					instance_feature = checkpoint(self.layers[i], instance_feature)
				else:
					instance_feature = self.layers[i](instance_feature)
			elif op == "deformable_ins":
				if self.use_checkpoint and self.training:
					instance_feature = checkpoint(self.layers[i],
						instance_feature,
						anchor,
						anchor_embed,
						feature_maps,
						metas,
					)
				else:
					instance_feature = self.layers[i](
						instance_feature,
						anchor,
						anchor_embed,
						feature_maps,
						metas,
					)
			elif op == "refine_ins":
				anchor, cls, qt = self.layers[i](
					instance_feature,
					anchor,
					anchor_embed,
					time_interval=time_interval,
					return_cls=True,
				)
				prediction.append(anchor)
				classification.append(cls)
				quality.append(qt)
				prediction_gs.append("no_gs")
				if len(prediction) == self.num_single_frame_decoder:
					instance_feature, anchor = self.instance_bank.update(
						instance_feature, anchor, cls
					)
				anchor_embed = self.anchor_encoder(anchor)
				gs_feature = (instance_feature.unsqueeze(2) + init_gs_feature.unsqueeze(1)).flatten(1,2) #[bs, n*k, dim]
				gs_anchor = gs_anchor.unsqueeze(1).repeat(1, num_anchor, 1, 1)
				gs_anchor_embed = self.gs_anchor_encoder(gs_anchor).flatten(1,2) #[bs, n*k, dim]
			elif op == "temp_gnn_ins":
				instance_feature = torch.mean(gs_feature.reshape(batch_size, num_anchor, self.num_kpts_per_ins, self.embed_dims), dim=2).squeeze(2) # [bs, n*k, dim] - mean - [bs, n, dim]
				instance_feature = self.graph_model( 
					i,
					instance_feature, # [bs, n, dim]
					temp_instance_feature,
					temp_instance_feature,
					query_pos=anchor_embed,
					key_pos=temp_anchor_embed,
					attn_mask=attn_mask
					if temp_instance_feature is None
					else None,
				)
			elif op == "gnn_ins":
				instance_feature = self.graph_model(
					i,
					instance_feature, 
					value=instance_feature,
					query_pos=anchor_embed,
					attn_mask=attn_mask,
				)
				gs_feature = instance_feature.unsqueeze(2).repeat(1,1,self.num_kpts_per_ins,1).flatten(1,2) + gs_feature
			elif op == "sparse_conv_gs":
				xyz = (anchor[..., None, :3] + gs_anchor[...,:3])
				kpt_anchor = torch.cat([xyz, gs_anchor[..., 3:]], dim=-1).flatten(1,2)
				if self.use_checkpoint and self.training:
					gs_feature = checkpoint(self.layers[i],
						gs_feature,
						kpt_anchor,
					)
				else:
					gs_feature = self.layers[i](
						gs_feature,
						kpt_anchor,
					)
			elif op == "deformable_gs": 
				kpt_anchor_embed = self.gs_anchor_encoder(kpt_anchor)
				if self.use_checkpoint and self.training:
					gs_feature = checkpoint(self.layers[i], #[bs, n*k, dim]
						gs_feature,
						kpt_anchor,
						kpt_anchor_embed,
						feature_maps,
						metas,
					)
				else:
					gs_feature = self.layers[i]( #[bs, n*k, dim]
						gs_feature,
						kpt_anchor,
						kpt_anchor_embed,
						feature_maps,
						metas,
					)
			elif op == "norm_gs" or op == "ffn_gs":
				if self.use_checkpoint and self.training:
					gs_feature = checkpoint(self.layers[i], gs_feature)
				else:
					gs_feature = self.layers[i](gs_feature)
			elif op == "refine_gs": 
				anchor, gs_anchor, cls, qt = self.layers[i](
					gs_feature,
					anchor,
					anchor_embed,
					gs_anchor.flatten(1,2),
					gs_anchor_embed,
					time_interval=time_interval,
				)
				classification.append(cls)
				prediction.append(anchor)
				quality.append(qt)
				prediction_gs.append(gs_anchor)
				anchor_embed = self.anchor_encoder(anchor)
				if (
					len(prediction) > self.num_single_frame_decoder
					and temp_anchor_embed is not None
				):
					temp_anchor_embed = anchor_embed[
						:, : self.instance_bank.num_temp_instances
					]
				gs_anchor_embed = self.gs_anchor_encoder(gs_anchor).flatten(1,2)
				instance_feature = torch.mean(gs_feature.reshape(batch_size, num_anchor, self.num_kpts_per_ins, self.embed_dims), dim=2).squeeze(2)

			elif op == "refine": 
				anchor, cls, qt = self.layers[i](
					gs_feature,
					anchor,
					anchor_embed,
					time_interval=time_interval,
				)
				classification.append(cls)
				prediction.append(anchor)
				quality.append(qt)
				prediction_gs.append("no_gs")
				if (
					len(prediction) > self.num_single_frame_decoder
					and temp_anchor_embed is not None
				):
					temp_anchor_embed = anchor_embed[
						:, : self.instance_bank.num_temp_instances
					]
				instance_feature = torch.mean(gs_feature.reshape(batch_size, num_anchor, self.num_kpts_per_ins, self.embed_dims), dim=2).squeeze(2)
			else:
				raise NotImplementedError(f"{op} is not supported.")

		output = {}

		output.update(
			{
				"classification": classification,
				"prediction": prediction,
				"quality": quality,
				"prediction_gs": prediction_gs,
				"instance_feature": instance_feature,
				"anchor_embed": anchor_embed,
			}
		)

		# cache current instances for temporal modeling
		self.instance_bank.cache(
			instance_feature, anchor, cls, metas, feature_maps
		)
		if self.with_instance_id:
			instance_id = self.instance_bank.get_instance_id(
				cls, anchor, self.decoder.score_threshold
			)
			output["instance_id"] = instance_id
		return output

	@force_fp32(apply_to=("model_outs"))
	def loss(self, model_outs, data, feature_maps=None):
		# ===================== prediction losses ======================
		cls_scores = model_outs["classification"]
		reg_preds = model_outs["prediction"]
		gs_preds = model_outs["prediction_gs"]
		quality = model_outs["quality"]
		output = {}
		for decoder_idx, (cls, reg, gs, qt) in enumerate(
			zip(cls_scores, reg_preds, gs_preds, quality)
		):
			reg = reg[..., : len(self.reg_weights)]
			if not self.with_gs_supervision or gs == "no_gs" or decoder_idx != 5:
				cls_target, reg_target, reg_weights = self.sampler.sample(
					cls,
					reg,
					data[self.gt_cls_key],
					data[self.gt_reg_key],
				)
			else:
				cls_target, reg_target, reg_weights, vx_target, mask_target = self.sampler.sample_with_gs(
					cls,
					reg,
					data[self.gt_cls_key],
					data[self.gt_reg_key],
					data[self.gt_occ_key],
					data[self.gt_camera_mask],
					data[self.gt_ins_id],
					self.crop_size,
				)
			reg_target = reg_target[..., : len(self.reg_weights)]
			mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))

			num_pos = max(
				reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
			)
			if self.cls_threshold_to_reg > 0:
				threshold = self.cls_threshold_to_reg
				mask = torch.logical_and(
					mask, cls.max(dim=-1).values.sigmoid() > threshold
				)

			cls = cls.flatten(end_dim=1)
			cls_target = cls_target.flatten(end_dim=1)
			cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

			mask = mask.reshape(-1)
			reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
			reg_target = reg_target.flatten(end_dim=1)[mask]
			reg = reg.flatten(end_dim=1)[mask]
			reg_weights = reg_weights.flatten(end_dim=1)[mask]
			reg_target = torch.where(
				reg_target.isnan(), reg.new_tensor(0.0), reg_target
			)
			cls_target = cls_target[mask]
			if qt is not None:
				qt = qt.flatten(end_dim=1)[mask]

			reg_loss = self.loss_reg(
				reg,
				reg_target,
				weight=reg_weights,
				avg_factor=num_pos,
				prefix=f"{self.task_prefix}_",
				suffix=f"_{decoder_idx}",
				quality=qt,
				cls_target=cls_target,
			)

			output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
			output.update(reg_loss)

			if self.with_gs_supervision and gs != "no_gs" and decoder_idx == 5:
				gs = gs.flatten(end_dim=1)[mask] #[n, k, 11]
				gs[..., :3] = gs[..., :3] + reg[...,:3].unsqueeze(1).repeat(1, self.num_kpts_per_ins, 1) - reg_target[...,:3].unsqueeze(1).repeat(1, self.num_kpts_per_ins, 1)
				gsgroup = gs.clone()
				gsgroup[..., :2] = torch.clamp(gs[..., :2], -self.clamp_h+1e-2, self.clamp_h-1e-2)
				gsgroup[..., 2:3] = torch.clamp(gs[..., 2:3], -self.clamp_d+1e-2, self.clamp_d-1e-2)
				vx, flag = self.gs2vx(gsgroup, gs.device)
				
				vx_target = vx_target.flatten(end_dim=1)[mask].to(torch.long)
				mask_target = mask_target.flatten(end_dim=1)[mask]

				if decoder_idx == 5:
					iou = self.iou_binary(vx, vx_target, mask_target).detach()
					output[f"{self.task_prefix}_iou_{decoder_idx}"] = iou

					iou_empty = self.iou_binary_empty(vx, vx_target, mask_target).detach()
					output[f"{self.task_prefix}_iou_empty_{decoder_idx}"] = iou_empty
				vx_loss = self.loss_vx(vx.flatten(), vx_target.flatten(), flag, gs, mask_target.flatten(), avg_factor=(((vx_target==1) & mask_target).sum().item() + 0.1 * ((vx_target==0) & mask_target).sum().item()))

				output[f"{self.task_prefix}_loss_vx_{decoder_idx}"] = vx_loss

		return output

	def gs2vx(self, gs_group, device):
		n, k, _ = gs_group.shape
		if n == 0:
			ins_vx = torch.empty(n, self.h, self.w, self.d, dtype=gs_group.dtype, device=gs_group.device)
			flag = True
		else:
			scales = gs_group[..., 3:6].clone()
			rotations = gs_group[..., 6:10].clone()
			S = torch.zeros(n, k, 3, 3, dtype=gs_group.dtype, device=gs_group.device)
			S[..., 0, 0] = scales[..., 0]
			S[..., 1, 1] = scales[..., 1]
			S[..., 2, 2] = scales[..., 2]
			R = self.get_rotation_matrix(rotations) # n, k, 3, 3
			M = torch.matmul(S, R)
			Cov = torch.matmul(M.transpose(-1, -2), M)
			CovInv = torch.inverse(Cov) # n, k, 3, 3

			centers = self.gen_centers(self.h, self.w, self.d, self.voxel_size).to(gs_group.device).flatten(end_dim=-2)
			centers[:,0] = centers[:,0]-self.h * self.voxel_size / 2
			centers[:,1] = centers[:,1]-self.w * self.voxel_size / 2
			centers[:,2] = centers[:,2]-self.d * self.voxel_size / 2

			flag = False
			ins_voxel = (
				gs_group.new_zeros([n, self.h, self.w, self.d], dtype=torch.float)
				)
			for i in range(n):
				gs = gs_group[i]
				covinv = CovInv[i]
				vx = self.splatting(centers, gs[:, :3], gs[:, 3:6], covinv).reshape(self.h, self.w, self.d).to(dtype=torch.float)
				ins_voxel[i] = vx
			if torch.isnan(ins_voxel).any() or torch.isinf(ins_voxel).any():
				torch.nan_to_num(ins_voxel, nan=0.0, posinf=0.0, neginf=0.0)
			ins_vx = torch.clamp(ins_voxel, 0, 0.999)
		return ins_vx, flag

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

	def iou_binary(self, pred, label, mask=None, EMPTY=1.):
		"""
		IoU for foreground class
		binary: 1 foreground, 0 background
		"""
		intersection = ((label == 1) & (pred >= 0.5) & mask).sum()
		union = (((label == 1) | (pred >= 0.5)) & mask).sum()
		if not union:
			iou = EMPTY
		else:
			iou = float(intersection) / float(union)
		
		return pred.new_tensor(100 * iou)

	def iou_binary_empty(self, pred, label, mask=None, EMPTY=1.):
		"""
		IoU for foreground class
		binary: 1 foreground, 0 background
		"""
		intersection = ((label == 0) & (pred < 0.5) & mask).sum()
		union = (((label == 0) | (pred < 0.5)) & mask).sum()
		if not union:
			iou = EMPTY
		else:
			iou = float(intersection) / float(union)
		
		return pred.new_tensor(100 * iou)

	@force_fp32(apply_to=("model_outs"))
	def post_process(self, model_outs, data, output_idx=-1):
		return self.decoder.decode(
			model_outs["classification"],
			model_outs["prediction"],
            model_outs["prediction_gs"],
            data,
			model_outs.get("instance_id"),
			model_outs.get("quality"),
			output_idx=output_idx,
		)
