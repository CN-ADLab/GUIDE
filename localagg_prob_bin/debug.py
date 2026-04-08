import torch
from local_aggregate_prob import LocalAggregator
import time

P = 25600
N = 80000

a = LocalAggregator(3, 200, 200, 16, [-40.0, -40.0, -1.0], 0.4).cuda()
s = torch.tensor([80.0, 80.0, 6.4]).cuda()
o = torch.tensor([-40.0, -40.0, -1.0]).cuda()

pts = torch.rand(N, 3).cuda() * s + o
means3D = torch.rand(P, 3).cuda() * s + o
# opas = torch.randn(P).sigmoid().cuda()
# semantics = torch.rand(P, 18).softmax(dim=-1).cuda()

scales = torch.rand(P, 3).cuda()
scales_mat = torch.diag_embed(scales)
cov3D = scales_mat.transpose(-1, -2) @ scales_mat
cov3D = cov3D.inverse()
# breakpoint()
# cov3D = torch.rand(1, P, 3, 3).cuda()

means3D.requires_grad_(True)
# opas.requires_grad_(True)
# semantics.requires_grad_(True)
cov3D.requires_grad_(True)

for i in range(10):
    s = time.perf_counter()
    out = a(pts, means3D, scales, cov3D)#.cpu()
    e = time.perf_counter()
    print(e-s)

# mi = out.min()
# ma = out.max()
# import pdb; pdb.set_trace()