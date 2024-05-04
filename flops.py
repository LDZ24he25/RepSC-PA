# from torchvision.models import resnet50
import torch
from thop import profile
# from torch_flops import TorchFLOPsByFX
import argparse
import random
from basicsr.models import build_model
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from os import path as osp
from collections import OrderedDict
import yaml
from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
# from odisr.archs.osrt_arch import SwinTransformerBlock,OSRT
# from odisr.archs.osrt_arch import OSRT
from odisr.archs.osrt_archx4 import OSRT
import torch
import torch.nn as nn
from torchsummaryX import summary
from basicsr.archs.arch_util import to_2tuple, trunc_normal_, flow_warp, DCNv2Pack
# from odisr.archs import osrt_archesrt
from ptflops import get_model_complexity_info
from pthflops import count_ops

class Connet1x1and1x1(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(Connet1x1and1x1, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(1, 1), stride=1,
                                         padding=(0,0), dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_1x1_branch1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0,0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0,0), dilation=1, groups=1, padding_mode='zeros')

    def forward(self, inputs):
        if (self.deploy):
            return self.activation(self.rbr_reparam(inputs))
        else:
            return self.activation(
                self.rbr_1x1_branch1(inputs) + self.rbr_1x1_branch2(inputs))
        
    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1),
                                        stride=1, padding=(0, 0), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # self.__delattr__('rbr_condition_branch')
        self.__delattr__('rbr_1x1_branch1')
        self.__delattr__('rbr_1x1_branch2')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
                # condition_branch
        # kernel_condition, bias_condition = self.rbr_condition_branch.weight.data, self.rbr_condition_branch.bias.data

        # 3x3 branch
        kernel_1 = self._fuse_1x1_1x1_branch(self.rbr_1x1_branch1,
                                                     self.rbr_1x1_branch2)

        # kernel_2, bias_2 = self.rbr_1x1_branch2.weight.data, self.rbr_1x1_branch2.bias.data

        return kernel_1 
    
    def _fuse_1x1_1x1_branch(self, conv1, conv2):
        weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
        return weight

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        act_func = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        act_func = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    # TODO: 新增silu和gelu激活函数
    elif act_type == 'silu':
        pass
    elif act_type == 'gelu':
        pass
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return act_func

class Concat1x1and3x3(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='lrelu', deploy=False):
        super(Concat1x1and3x3, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(1, 1), stride=1,
                                         padding=(0,0), dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0,0), dilation=1, groups=1, padding_mode='zeros')

    def forward(self, inputs):
        if (self.deploy):
            return self.activation(self.rbr_reparam(inputs))
        else:
            return self.activation(
                self.rbr_3x3_branch(inputs) + self.rbr_1x1_branch(inputs))
        
    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,1),
                                        stride=1, padding=(0, 0), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # self.__delattr__('rbr_condition_branch')
        self.__delattr__('rbr_3x3_branch')
        self.__delattr__('rbr_1x1_branch')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
                # condition_branch
        # kernel_condition, bias_condition = self.rbr_condition_branch.weight.data, self.rbr_condition_branch.bias.data

        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data

        kernel_1x1, bias_1x1 = self.rbr_1x1_branch.weight.data, self.rbr_1x1_branch.bias.data
        # 1x1 1x3 3x1 branch
        # kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
        #                                                                                self.rbr_1x3_branch,
        #                                                                                self.rbr_3x1_branch)
        # 1x1+3x3 branch
        # kernel_1x1_3x3_fuse = self._fuse_1x1_3x3_branch(self.rbr_1x1_3x3_branch_1x1,
        #                                                 self.rbr_1x1_3x3_branch_3x3)
        # identity branch
        # device = kernel_1x1_1x3_3x1_fuse.device  # just for getting the device
        # kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        # for i in range(self.out_channels):
        #     kernel_identity[i, i, 1, 1] = 1.0

        # kernel_1x1_sbx, bias_1x1_sbx = self.rbr_conv1x1_sbx_branch.rep_params()
        # kernel_1x1_sby, bias_1x1_sby = self.rbr_conv1x1_sby_branch.rep_params()
        # kernel_1x1_lpl, bias_1x1_lpl = self.rbr_conv1x1_lpl_branch.rep_params()

        return kernel_3x3 + kernel_1x1 , bias_3x3 + bias_1x1

class FFT(nn.Module):
    def __init__(self, dim):
        super(FFT, self).__init__()
        self.conv1 = nn.Conv2d(dim*2, dim*2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim*2, dim*2, 1, 1, 0)
        self.relu1 = nn.ReLU()
        self.conv_change  = nn.Conv2d(dim*2, dim,1,1,0)

        # self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.conv4 = nn.Conv2d(dim, dim, 3, 1, 1)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        # _, _, H, W = x.shapes
        
        y = torch.fft.fft2(x, dim=(-2,-1))
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat((y_real, y_imag), dim=1)
        # print(y_f.shape, '1111111111')
        y_f = self.conv1(y_f)
        y_f = self.relu1(y_f)
        y_f = self.conv2(y_f)
        # y_real, y_imag = torch.chunk(y_f, 2, dim=1)
        # y = torch.complex(y_real, y_imag)
        y = torch.fft.ifft2(y_f,dim=(-2,-1))
        y = self.conv_change(y.real)
        # print(y.shape,'100000')
    #   x_fft = torch.fft.ifft2(torch.complex(x_fft_sigmoid[..., 0], x_fft_sigmoid[..., 1]), dim=(-2, -1))

        # y_123 = self.conv3(x)
        # y_123 = self.relu2(y_123)
        # y_123 = self.conv4(y_123)
        # print(type(y_123 + x + y),'00000000000000')
        return y
        # return y_123 + x + y

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super(UpsampleOneStep, self).__init__()
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        # m = []
        # m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        # m.append(nn.PixelShuffle(scale))
        self.conv1 = nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1)
        self.pixelshuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixelshuffle(x)
        return x
       
    # inputs = torch.zeros((100, 1), dtype=torch.long)  # [length, batch_size]
    # summary(Net(), inputs)

    # root = "/media/Storage3/ldz/OSRT-master/options/test/OSRT_light_x2_fisheye_aug.yml"
    # opt, _ = parse_options(root, is_train=False)
    # model = build_model(opt)
# model = osrt_arch.OSRT(img_size=64,
#                  patch_size=1,
#                  in_chans=3,
#                  embed_dim=96,
#                  depths=(6, 6, 6, 6),
#                  num_heads=(6, 6, 6, 6),
#                  window_size=7,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm,
#                  ape=False,
#                  patch_norm=True,
#                  use_checkpoint=False,
#                  upscale=2,
#                  img_range=1.,
#                  upsampler='',
#                  resi_connection='1conv',
#                  condition_dim=1,
#                  vit_condition=None,
#                  vit_condition_type='1conv',
#                  dcn_condition=None,
#                  dcn_condition_type='1conv',
#                  window_condition=False,
#                  window_condition_only=False,
#                  c_dim=60)

# model = osrt_arch.SCPA(60, reduction=2, stride=1, dilation=1)
# model = osrt_arch.Upsamplelightweight(scale=2, num_feat=60, num_out_ch=3, input_resolution=None)
# model = UpsampleOneStep(scale=2, num_feat=60, num_out_ch=3, input_resolution=None)
# model = osrt_arch.RSTB(
#                  60,
#                  input_resolution=[63,63],
#                  depth=6,
#                  num_heads=6,
#                  window_size=7,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  norm_layer=nn.LayerNorm,
#                  downsample=None,
#                  use_checkpoint=False,
#                  img_size=224,
#                  patch_size=4,
#                  resi_connection='1conv',
#                  vit_condition=0,
#                  vit_condition_type='1conv',
#                  condition_dim=1,
#                  use_dcn_conv=False,
#                  dcn_condition_type='1conv',
#                  window_condition=False,
#                  window_condition_only=False,
#                  c_dim=60,
#                  )
# checkpoints = '/media/Storage3/ldz/OSRT-master/experiments/OSRT_light_x2_fisheye_noaug/models/net_g_585000.pth'
# model = torch.load(checkpoints)
# model_name = 'yolov3 cut asff'
# model = osrt_arch.FAM(in_channels=60, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False, split=2, reduction=2)
# model = osrt_archesrt.ESDB(in_channels=60, out_channels=60, conv=BSConvU, p=0.25)
# model = SwinTransformerBlock(dim=60,
#                  input_resolution=[64,64],
#                  num_heads=6,
#                  window_size=8,
#                  shift_size=0,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                 #  conv_scale=0.01,
#                  compress_ratio=3,
#                  squeeze_factor=30,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm,
#                  use_vit_condition=False,
#                  condition_dim=1,
#                  vit_condition_type='1conv',
#                  window_condition=False,
#                  window_condition_only=False,
#                  c_dim=60,)
# class offset(nn.Module):
#     def __init__(self,c_dim, dim):
#         super(offset, self).__init__()
#         self.offset_conv = nn.Sequential(nn.Conv2d(1, c_dim, 1, 1, 0, bias=True),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                  nn.Conv2d(c_dim, dim, 1, 1, 0, bias=True),
#                                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
#     def forward(self,x):
#         x = self.offset_conv(x)
#         return x
    
class dcn(nn.Module):
    def __init__(self, dim):
        super(dcn, self).__init__()
        self.dcn = DCNv2Pack(dim, dim, 3, padding=1)
        
    def forward(self,x,c):
        x = self.dcn(x, c)
        return x

model = OSRT(upscale=2,
        img_size=(64, 64),
        window_size=8,
        img_range=1.,
        depths=[1],
        embed_dim=30,
        c_dim=30,
        num_heads=[6],
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
x = torch.randn(1, 3, 64, 64)
x_size = (64, 64)
c = torch.randn(1, 1, 64, 64)
# summary(model, x)

def prepare_input(resolution):
    x = torch.FloatTensor(1, 3, 64, 64)
    condition = torch.FloatTensor(1, 1, 64, 64)
    return dict(x = x, condition = condition)


# count_ops(model, (x,c))
# prepare_input = prepare_input(1)

# flops, params = profile(model, inputs=(x,c))

# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# 
macs, params = get_model_complexity_info(model, input_res=((1, 3, 64, 64),(1, 1, 64, 64)), 
                                        input_constructor=prepare_input,
                                        as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# flops, params = profile(model, inputs=(input, ),verbose=True)
# print("%s | %.2f | %.2f" % (model_name, params / (1000 ** 2), flops / (1000 ** 3)))#这里除以1000的平方，是为了化成M的单位

# flops_counter = TorchFLOPsByFX(model)

# flops_counter.propagate(x, c)

# total_flops = flops_counter.print_total_flops(show=True)
# print("=" * 10, "vit_base16", "=" * 10)
#     flops_counter = TorchFLOPsByFX(vit)
#     # # Print the grath (not essential)
#     # print('*' * 120)
#     # flops_counter.graph_model.graph.print_tabular()
#     # Feed the input tensor
#     flops_counter.propagate(x)
#     # # Print the flops of each node in the graph. Note that if there are unsupported operations, the "flops" of these ops will be marked as 'not recognized'.
#     # print('*' * 120)
#     # flops_counter.print_result_table()
#     # # Print the total FLOPs
#     total_flops = flops_counter.print_total_flops(show=True)

#     print("=" * 10, "resnet50", "=" * 10)
#     flops_counter = TorchFLOPsByFX(resnet)
#     flops_counter.propagate(x)
#     total_flops = flops_counter.print_total_flops(show=True)
# import torch
# import torch.nn as nn
# # from torchstat import stat
# from thop import profile

# class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 2, kernel_size=7, stride=2, padding=3, bias=False)
#    def forward(self, x):
#        x = self.conv1(x)

# net = Net()
# # stat(net, (3, 500, 500))

# input = torch.randn(1, 3, 500, 500)
# flops, params = profile(net, inputs=(input,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')