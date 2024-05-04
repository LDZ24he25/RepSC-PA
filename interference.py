import os
import numpy as np
import torch
# from torchvision.models import resnet18
import time
from odisr.archs.osrt_archx4 import OSRT

if __name__ == '__main__':
    model = OSRT(upscale=2,
        img_size=(64, 64),
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        c_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',
        # resi_connection='1conv',
        condition_dim=1,
        vit_condition=[6, 6, 6, 6],
        vit_condition_type='3conv',
        dcn_condition=[1, 1, 1, 1, 1],
        dcn_condition_type='2conv',
        window_condition=True,
            )
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    dump_input = torch.ones(1,3,64,64).to(device)

    c = torch.ones(1, 1, 64, 64).to(device)

    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input, c)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))
        print(torch.cuda.max_memory_allocated())

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(dump_input, c)
    print(prof.table())

