import torch

# def make_coord(shape, ranges=(-1, 1), flatten=False):
#     """ Make coordinates at grid centers.
#     """
#     coord_seqs = []
#     for i, n in enumerate(shape):
#         v0, v1 = ranges
#         r = (v1 - v0) / (2 * n)
#         seq = v0 + r + (2 * r) * torch.arange(n).float()
#         coord_seqs.append(seq)
#     ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
#     if flatten:
#         ret = ret.view(-1, ret.shape[-1])
#     return ret

coord_seqs = []

for i, n in enumerate([7]):
    print(i,n)

    v0, v1 = (-1, 1)
    r = (v1 -v0) / (2* n)
    print(n)
    seq = v0 + r + (2*r) *torch.arange(n).float()
    print(seq)
    coord_seqs.append(seq)

ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
print(torch.arange(7).float())

print(ret)


