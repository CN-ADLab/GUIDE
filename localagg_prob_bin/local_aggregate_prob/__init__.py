#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C

import copy
import pickle
import warnings


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        # opas,
        # semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            # opas,
            # semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, bin_logits, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            pts,
            points_int,
            cov3D,
            # opas,
            # semantics,
            # logits,
            bin_logits,
            # density,
            # probability
        )
        return bin_logits

    @staticmethod # todo
    def backward(ctx, bin_logits_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, bin_logits = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            # opas,
            # semantics,
            # logits,
            bin_logits,
            # density,
            # probability,
            # logits_grad,
            bin_logits_grad,
            # density_grad
            )

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            # opas_grad,
            # semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.pc_range_min = torch.tensor(pc_min, dtype=torch.float).unsqueeze(0)
        self.grid_size = grid_size
        self.radii_min = radii_min

    def forward(
        self, 
        pts,
        means3D, 
        # opas,
        # semantics, 
        scales, 
        cov3D): 

        # assert pts.shape[0] == 1
        # pts = pts#.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D#.squeeze(0)
        # opas = opas#.squeeze(0)
        # semantics = semantics#.squeeze(0)
        scales = scales.detach()#.squeeze(0)
        cov3D = cov3D#.squeeze(0)

        points_int = ((pts - self.pc_range_min.to(pts.device)) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_range_min.to(pts.device)) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # input_o = dict(
        #     means3D = means3D,
        #     radii = radii,
        #     cov3D = cov3D,
        # )
        # input_o = copy.deepcopy(self.tensors_to_numpy(input_o))

        # try:
        # Invoke C++/CUDA rasterization routine
        bin_logits = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            # opas,
            # semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )
        # except Exception as e:
        #     with open('prob.pkl', 'wb') as f:
        #         pickle.dump(input_o, f)
        #     warnings.warn(f'\n\nencounter fatal error, debug info saved to prob.pkl !!\n')
        #     raise e

        return bin_logits # n, c; n, c; n

    # def tensors_to_numpy(obj):
    #     if isinstance(obj, torch.Tensor):
    #         return obj.detach().cpu().numpy()
    #     elif isinstance(obj, dict):
    #         return {k: tensors_to_numpy(v) for k, v in obj.items()}
    #     elif isinstance(obj, list):
    #         return [tensors_to_numpy(v) for v in obj]
    #     else:
    #         return obj