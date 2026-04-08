import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES

from projects.mmdet3d_plugin.core.box3d import *

from torch import Tensor
from typing import Callable, Optional
import torch.nn.functional as F

@LOSSES.register_module()
class SparseBox3DLoss(nn.Module):
	def __init__(
		self,
		loss_box,
		loss_centerness=None,
		loss_yawness=None,
		cls_allow_reverse=None,
	):
		super().__init__()

		def build(cfg, registry):
			if cfg is None:
				return None
			return build_from_cfg(cfg, registry)

		self.loss_box = build(loss_box, LOSSES)
		self.loss_cns = build(loss_centerness, LOSSES)
		self.loss_yns = build(loss_yawness, LOSSES)
		self.cls_allow_reverse = cls_allow_reverse

	def forward(
		self,
		box,
		box_target,
		weight=None,
		avg_factor=None,
		prefix="",
		suffix="",
		quality=None,
		cls_target=None,
		**kwargs,
	):
		# Some categories do not distinguish between positive and negative
		# directions. For example, barrier in nuScenes dataset.
		if self.cls_allow_reverse is not None and cls_target is not None:
			if_reverse = (
				torch.nn.functional.cosine_similarity(
					box_target[..., [SIN_YAW, COS_YAW]],
					box[..., [SIN_YAW, COS_YAW]],
					dim=-1,
				)
				< 0
			)
			if_reverse = (
				torch.isin(
					cls_target, cls_target.new_tensor(self.cls_allow_reverse)
				)
				& if_reverse
			)
			box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
				if_reverse[..., None],
				-box_target[..., [SIN_YAW, COS_YAW]],
				box_target[..., [SIN_YAW, COS_YAW]],
			)

		output = {}
		box_loss = self.loss_box(
			box, box_target, weight=weight, avg_factor=avg_factor
		)
		output[f"{prefix}loss_box{suffix}"] = box_loss

		if quality is not None:
			cns = quality[..., CNS]
			yns = quality[..., YNS].sigmoid()
			cns_target = torch.norm(
				box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
			)
			cns_target = torch.exp(-cns_target)
			cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
			output[f"{prefix}loss_cns{suffix}"] = cns_loss

			yns_target = (
				torch.nn.functional.cosine_similarity(
					box_target[..., [SIN_YAW, COS_YAW]],
					box[..., [SIN_YAW, COS_YAW]],
					dim=-1,
				)
				> 0
			)
			yns_target = yns_target.float()
			yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
			output[f"{prefix}loss_yns{suffix}"] = yns_loss
		return output

@LOSSES.register_module()
class VXFocalLoss(nn.Module):

	def __init__(self,
				 use_sigmoid=True,
				 gamma=2.0,
				 alpha=0.25,
				 reduction='mean',
				 loss_weight=1.0,
				 activated=False):
		"""`Focal Loss <https://arxiv.org/abs/1708.02002>`_

		Args:
			use_sigmoid (bool, optional): Whether to the prediction is
				used for sigmoid or softmax. Defaults to True.
			gamma (float, optional): The gamma for calculating the modulating
				factor. Defaults to 2.0.
			alpha (float, optional): A balanced form for Focal Loss.
				Defaults to 0.25.
			reduction (str, optional): The method used to reduce the loss into
				a scalar. Defaults to 'mean'. Options are "none", "mean" and
				"sum".
			loss_weight (float, optional): Weight of loss. Defaults to 1.0.
			activated (bool, optional): Whether the input is activated.
				If True, it means the input has been activated and can be
				treated as probabilities. Else, it should be treated as logits.
				Defaults to False.
		"""
		super(VXFocalLoss, self).__init__()
		# assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
		self.use_sigmoid = use_sigmoid
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction
		self.loss_weight = loss_weight
		self.activated = activated

	def forward(self,
				pred,
				target,
				flag,
				orig_pred,
				mask_target,
				weight=None,
				avg_factor=None,
				reduction_override=None):
		"""Forward function.

		Args:
			pred (torch.Tensor): The prediction.
			target (torch.Tensor): The learning label of the prediction.
				The target shape support (N,C) or (N,), (N,C) means
				one-hot form.
			weight (torch.Tensor, optional): The weight of loss for each
				prediction. Defaults to None.
			avg_factor (int, optional): Average factor that is used to average
				the loss. Defaults to None.
			reduction_override (str, optional): The reduction method used to
				override the original reduction method of the loss.
				Options are "none", "mean" and "sum".

		Returns:
			torch.Tensor: The calculated loss
		"""
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)
		

		assert pred.dim() == target.dim()

		if flag:
			return (orig_pred.flatten().sum() * 0).mean()
		
		calculate_loss_func = py_focal_loss
				

		loss_cls = self.loss_weight * calculate_loss_func(
			pred,
			target,
			mask_target,
			weight,
			gamma=self.gamma,
			alpha=self.alpha,
			reduction=reduction,
			avg_factor=avg_factor)

		
		return loss_cls

def weight_reduce_loss(loss: Tensor,
					   mask: Tensor,
					   weight: Optional[Tensor] = None,
					   reduction: str = 'mean',
					   avg_factor: Optional[float] = None) -> Tensor:
	# if weight is specified, apply element-wise weight
	if weight is not None:
		loss = loss * weight

	# if avg_factor is not specified, just reduce the loss
	if avg_factor is None:
		loss = reduce_loss(loss, reduction)
	else:
		# if reduction is mean, then average the loss by avg_factor
		if reduction == 'mean':
			# Avoid causing ZeroDivisionError when avg_factor is 0.0,
			# i.e., all labels of an image belong to ignore index.
			loss = torch.where(mask, loss, 0)
			eps = torch.finfo(torch.float32).eps
			loss = loss.sum() / (avg_factor + eps)
		# if reduction is 'none', then do nothing, otherwise raise an error
		elif reduction != 'none':
			raise ValueError('avg_factor can not be used with reduction="sum"')
	return loss

def py_focal_loss(pred_sigmoid,
					target,
					mask,
					weight=None,
					gamma=2.0,
					alpha=0.25,
					reduction='mean',
					avg_factor=None):
	# pred_sigmoid = pred.sigmoid()
	target = target.type_as(pred_sigmoid)
	# Actually, pt here denotes (1 - pt) in the Focal Loss paper
	pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
	# Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
	focal_weight = (alpha * target + (1 - alpha) *
					(1 - target)) * pt.pow(gamma)
	loss = F.binary_cross_entropy(
		pred_sigmoid, target, reduction='none') * focal_weight
	if weight is not None:
		if weight.shape != loss.shape:
			if weight.size(0) == loss.size(0):
				# For most cases, weight is of shape (num_priors, ),
				#  which means it does not have the second axis num_class
				weight = weight.view(-1, 1)
			else:
				# Sometimes, weight per anchor per class is also needed. e.g.
				#  in FSAF. But it may be flattened of shape
				#  (num_priors x num_class, ), while loss is still of shape
				#  (num_priors, num_class).
				assert weight.numel() == loss.numel()
				weight = weight.view(loss.size(0), -1)
		assert weight.ndim == loss.ndim
	loss = weight_reduce_loss(loss, mask, weight, reduction, avg_factor)
	return loss