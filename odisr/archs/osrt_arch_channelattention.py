# Modified from https://github.com/JingyunLiang/SwinIR
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_, flow_warp, DCNv2Pack

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class LTE(nn.Module):

    def __init__(self, imnet_spec=None, hidden_dim=256):
        super().__init__()        
        # self.encoder = models.make(encoder_spec)
        # self.feat_coord = feat_coord
        self.coef = nn.Conv2d(96, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(96, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)        

        # self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def gen_feat(self, inp, feat_coord):
        self.inp = inp
        self.feat_coord = feat_coord
        
        # self.feat = self.encoder(inp)
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        coef = self.coeff
        freq = self.freqq

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1).contiguous()
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1).contiguous()
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1).contiguous()
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                
                # basis generation
                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)

                inp = torch.mul(q_coef, q_freq)            

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1).contiguous()
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


class FAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(FAM, self).__init__()
        self.gavepool = nn.AdaptiveAvgPool2d((None, None))
        self.avepool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)

        self.para1 = torch.nn.Parameter(torch.ones(1))
        self.para2 = torch.nn.Parameter(torch.zeros(1))
        self.para3 = torch.nn.Parameter(torch.ones(1))
        self.para4 = torch.nn.Parameter(torch.zeros(1))
            #     self.relative_position_bias_table = nn.Parameter(
            # torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
    

        # self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1, padding=0)
        # self.conv4 = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_g = self.gavepool(x)
        x_a = self.avepool(x)

        x_g_sub = x_g.sub(x)
        x_a_sub = x_a.sub(x)

        # x_g = self.conv1(x_g)
        # x_g_sub = self.conv2(x_g_sub)
        # x_a = self.conv3(x_a)
        # x_a_sub = self.conv4(x_a_sub)

        # x_g = torch.matmul(x_g, self.para1)
        # x_g_sub = torch.matmul(x_g_sub, self.para2)
        # x_a = torch.matmul(x_a, self.para3)
        # x_a_sub = torch.matmul(x_a_sub, self.para4)
        x_g = x_g * self.para1
        x_g_sub = x_g_sub * self.para2
        x_a = x_a * self.para3
        x_a_sub = x_a_sub * self.para4

        return x_g + x_g_sub + x_a + x_a_sub

# sam非常适合视频数据， 它可以在batchsize中寻找相互之间的关系
class SAM(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(SAM, self).__init__()
        self.batch_size = 20
        # self.in_split = []
        self.gavepool = nn.AdaptiveAvgPool2d((3, 3))
        self.conv_gave1 = nn.Conv2d(60,20, 1,1,0)
        self.conv_gave2 = nn.Conv2d(60,20, 1,1,0)
        self.conv_gave3 = nn.Conv2d(60,20, 1,1,0)
        self.para1 = torch.nn.Parameter(torch.ones(1))
        self.para2 = torch.nn.Parameter(torch.ones(1))
        self.para3 = torch.nn.Parameter(torch.ones(1))
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.conv_out = nn.Conv2d(60, 60, 1,1, 0)

        # c = torch.nn.Conv2d(1, 1, (3, 3), stride=2, padding=1, bias=False)
        # c.weight.data = torch.Tensor([[[[1, 1, 1],
        #                         [1, 1, 0],
        #                         [0, 1, 1]]]])
        
        # a = torch.Tensor([[[[1, 0, 0, 1, 2],
        #             [0, 2, 0, 0, 0],
        #             [1, 1, 0, 1, 0],
        #             [1, 0, 2, 2, 2],
        #             [1, 0, 0, 2, 0]]]])
        #     print(c(a))
        

    def blur(self, x):
        # kernel 
        # print(x.shape,'00000000000000000000')   # [1, 60, 512, 1024]
        x_gap = self.gavepool(x)

        x_gap1 = self.conv_gave1(x_gap)
        x_gap2 = self.conv_gave2(x_gap)
        x_gap3 = self.conv_gave3(x_gap)

        x_gap1 = x_gap1 * self.para1
        x_gap2 = x_gap2 * self.para2
        x_gap3 = x_gap3 * self.para3

        x_gap1 = self.tanh1(x_gap1)
        x_gap2 = self.tanh2(x_gap2)
        x_gap3 = self.tanh3(x_gap3)

        # print(x_gap1.shape, '10000000000000')
        # 一种方法是对卷积权重进行复制，另一种方法对input进                                                                                                                                                                                             行batchsize的复制，但是第二种不确定gpu是否能够容纳下
        # if self.training is False:
        #     x_gap1 = x_gap1.repeat(20, 1, 1, 1)
        #     x_gap2 = x_gap2.repeat(20, 1, 1, 1)
        #     x_gap3 = x_gap3.repeat(20, 1, 1, 1)

        x1 = F.conv2d(x[:,0:20,:,:].contiguous(), x_gap1, bias=None, stride=1, padding=1)
        # print(x1.shape, "10000000000000")
        # self.weight2 = torch.nn.Parameter(x_gap2, requires_grad=True)
        # self.weight2 = x_gap2
        x2 = F.conv2d(x[:,20:40,:,:].contiguous(), x_gap2, bias=None, stride=1, padding=1)

        # self.weight3 = torch.nn.Parameter(x_gap3, requires_grad=True)
        # self.weight3 = x_gap3
        x3 = F.conv2d(x[:,40:60,:,:].contiguous(), x_gap3, bias=None, stride=1, padding=1)

        x1 = torch.cat((x1, x2),dim=1)
        x1 = torch.cat((x1, x3),dim=1)
        # print(x1.shape,'1111111111111')    # [1, 3, 512, 1024]
        x1 = self.conv_out(x1) 
        return x1
    
    def forward(self, x):
        
        return self.blur(x)

class MAConv2(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAConv2, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'
        self.in_channels = in_channels
        self.num_split = split
        # self.dcn = DCNv2Pack(in_channels, in_channels, 3, padding=1)
        
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []
        # self.conv_delte = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=1, stride=1, padding=0)
        # self.conv_delte_2 = nn.Conv2d(in_channels=60, out_channels=15, kernel_size=1, stride=1, padding=0)

        self.offset_conv_1 = nn.Sequential(nn.Conv2d(1, 60, 1, 1, 0, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(60, 30, 1, 1, 0, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.offset_conv_2 = nn.Sequential(nn.Conv2d(1, 60, 1, 1, 0, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(60, 15, 1, 1, 0, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)

            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            # setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                # DCNv2Pack(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
                #           kernel_size=1,padding=0),
                # nn.ReLU(inplace=True),
                # DCNv2Pack(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
                #           kernel_size=1, padding=0),
            # ]))
            
            # self.dcn1 = DCNv2Pack(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
            #               kernel_size=1,padding=0)
            
            # self.relu1 = nn.ReLU(inplace=True)

            # self.dcn2 = DCNv2Pack(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
            #               kernel_size=1, padding=0)
            
            setattr(self, 'dcn1{}'.format(i), DCNv2Pack(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
                          kernel_size=1,padding=0))
            setattr(self, 'relu{}'.format(i), nn.ReLU(inplace=True))
            setattr(self, 'dcn2{}'.format(i), DCNv2Pack(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
                          kernel_size=1, padding=0))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split, 
                                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            # setattr(self, 'dcn1{}'.format(i), nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
            #               kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
            # setattr(self, 'relu{}'.format(i), nn.ReLU(inplace=True))
            # setattr(self, 'dcn2{}'.format(i), nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
            #               kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
            # setattr(self, 'conv{}'.format(i), DCNv2Pack(in_channels=in_split, out_channels=out_split, 
            #                                             kernel_size=1, stride=1, padding=0)) 

    def forward(self, input, condition):
        input = torch.split(input, self.in_split, dim=1)
        
        output = []
        condition_1 = self.offset_conv_1(condition)
        condition_2 = self.offset_conv_2(condition)
        for i in range(self.num_split):
            # scale, translation = torch.split((getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1), condition)),
            #                                  (self.in_split[i], self.in_split[i]), dim=1)
            input_1 = torch.cat(input[:i] + input[i + 1:], 1)
            # input_1 = getattr(self, 'dcn1{}'.format(i))(input_1, condition_1)
            # input_1 = getattr(self, 'relu{}'.format(i))(input_1)
            # input_1 = getattr(self, 'dcn2{}'.format(i))(input_1, condition_2)

            input_1 = getattr(self, 'dcn1{}'.format(i))(input_1, condition_1)
            input_1 = getattr(self, 'relu{}'.format(i))(input_1)
            input_1 = getattr(self, 'dcn2{}'.format(i))(input_1, condition_2)
            scale ,translation = torch.split(input_1, (self.in_split[i], self.in_split[i]), dim=1)
            # 单改最后一层conv
            
            output.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)

class MAConv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAConv, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'
        self.in_channels = in_channels
        self.num_split = split
        # self.dcn = DCNv2Pack(in_channels, in_channels, 3, padding=1)
        
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []
        # self.conv_delte = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=1, stride=1, padding=0)
        # self.conv_delte_2 = nn.Conv2d(in_channels=60, out_channels=15, kernel_size=1, stride=1, padding=0)

        self.offset_conv_1 = nn.Sequential(nn.Conv2d(1, 30, 1, 1, 0, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(30, 30, 1, 1, 0, bias=True),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.offset_conv_2 = nn.Sequential(nn.Conv2d(60, 60, 1, 1, 0, bias=True),
        #                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                             nn.Conv2d(60, 15, 1, 1, 0, bias=True),
        #                             nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)

            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            # setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                # DCNv2Pack(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
                #           kernel_size=1,padding=0),
                # nn.ReLU(inplace=True),
                # DCNv2Pack(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
                #           kernel_size=1, padding=0),
            # ]))
            
            # self.dcn1 = DCNv2Pack(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
            #               kernel_size=1,padding=0)
            
            # self.relu1 = nn.ReLU(inplace=True)

            # self.dcn2 = DCNv2Pack(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
            #               kernel_size=1, padding=0)
            
            # setattr(self, 'dcn1{}'.format(i), DCNv2Pack(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
            #               kernel_size=1,padding=0))
            # setattr(self, 'relu{}'.format(i), nn.ReLU(inplace=True))
            # setattr(self, 'dcn2{}'.format(i), DCNv2Pack(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
            #               kernel_size=1, padding=0))
            # setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split, 
            #                                             kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            setattr(self, 'dcn1{}'.format(i), nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
            setattr(self, 'relu{}'.format(i), nn.ReLU(inplace=True))
            setattr(self, 'dcn2{}'.format(i), nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=True))
            setattr(self, 'conv{}'.format(i), DCNv2Pack(in_channels=in_split, out_channels=out_split, 
                                                        kernel_size=1, stride=1, padding=0)) 

    def forward(self, input, condition):
        input = torch.split(input, self.in_split, dim=1)
        
        output = []
        condition_1 = self.offset_conv_1(condition)
        # condition_2 = self.offset_conv_2(condition)
        for i in range(self.num_split):
            # scale, translation = torch.split((getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1), condition)),
            #                                  (self.in_split[i], self.in_split[i]), dim=1)
            input_1 = torch.cat(input[:i] + input[i + 1:], 1)
            # input_1 = getattr(self, 'dcn1{}'.format(i))(input_1, condition_1)
            # input_1 = getattr(self, 'relu{}'.format(i))(input_1)
            # input_1 = getattr(self, 'dcn2{}'.format(i))(input_1, condition_2)

            input_1 = getattr(self, 'dcn1{}'.format(i))(input_1)
            input_1 = getattr(self, 'relu{}'.format(i))(input_1)
            input_1 = getattr(self, 'dcn2{}'.format(i))(input_1)
            scale ,translation = torch.split(input_1, (self.in_split[i], self.in_split[i]), dim=1)
            # 单改最后一层conv
            
            output.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(scale) + translation, condition_1))

        return torch.cat(output, 1)
    
class MAConv_origin(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAConv_origin, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
                          kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
                          kernel_size=1, stride=1, padding=0, bias=True),
            ]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split, 
                                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, input):
        input = torch.split(input, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            scale, translation = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1)),
                                             (self.in_split[i], self.in_split[i]), dim=1)
            output.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)

class MABlock(nn.Module):
    ''' Residual block based on MAConv '''
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 split=2, reduction=2):
        super(MABlock, self).__init__()

        # self.res = nn.Sequential(*[
        #     MAConv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction),
        #     nn.ReLU(inplace=True),
        #     MAConv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction),
        # ])

        self.maconv1 = MAConv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.maconv2 = MAConv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction)

    def forward(self, x, condition):

        return x + self.maconv2(self.relu(self.maconv1(x, condition)), condition)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xq, xkv=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if xkv is None:
            xkv = xq
        b_, n, c = xq.shape
        q = self.q(xq).reshape(b_, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        kv = self.kv(xkv).reshape(b_, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = q[0], kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        # calculate flops for 1 window with token length of n
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 conv_scale=0.01,
                 compress_ratio=3,
                 squeeze_factor=30,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_vit_condition=False,
                 condition_dim=1,
                 vit_condition_type='1conv',
                 window_condition=False,
                 window_condition_only=False,
                 c_dim=60,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.conv_scale = conv_scale
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        # channel attention block
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # add or concat 
        self.add_maconv = MAConv2(dim, dim, 1, 1, 0, bias=True)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

        self.use_vit_condition=use_vit_condition
        self.window_condition=window_condition
        self.window_condition_only=window_condition_only

        if use_vit_condition:
            if self.window_condition:
                if self.window_condition_only:
                    condition_dim = 2
                else:
                    condition_dim += 2

            if vit_condition_type == '1conv':
                self.offset_conv = nn.Sequential(
                    nn.Conv2d(condition_dim, 2, kernel_size=1, stride=1, padding=0, bias=True))
            elif vit_condition_type == '2conv':
                self.offset_conv = nn.Sequential(nn.Conv2d(condition_dim, c_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(c_dim, 2, 1, 1, 0, bias=True))
            elif vit_condition_type == '3conv':
                self.offset_conv = nn.Sequential(nn.Conv2d(condition_dim, c_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(c_dim, c_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(c_dim, 2, 1, 1, 0, bias=True))

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, condition):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        # calculate maconv
        

        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # channel attention block
        conv_x = self.conv_block(x.permute(0, 3, 1, 2).contiguous())
        conv_x = conv_x.permute(0,2,3,1).contiguous().view(b,h*w,c)

        x1 = x.view(b, c, h, w).contiguous()
        x_maconv = self.add_maconv(x1,condition)

        if self.use_vit_condition:
            x = x.permute(0, 3, 1, 2).contiguous()
            if self.window_condition:
                condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,8),torch.linspace(-1,1,8)))\
                    .type_as(x).unsqueeze(0).repeat(b, 1, h//self.window_size, w//self.window_size)
                if self.shift_size > 0:
                    condition_wind = torch.roll(condition_wind, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
                if self.window_condition_only:
                    _condition = condition_wind
                else:
                    _condition = torch.cat([condition, condition_wind], dim=1)
            else:
                _condition = condition
            offset = self.offset_conv(_condition).permute(0,2,3,1).contiguous()
            x_warped = flow_warp(x, offset, interp_mode='bilinear', padding_mode='border')
            x = x.permute(0, 2, 3, 1).contiguous()
            x_warped = x_warped.permute(0, 2, 3, 1).contiguous()

            # cyclic shift
            if self.shift_size > 0:
                shifted_x_warped = torch.roll(x_warped, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x_warped = x_warped

            # partition windows
            x_windows_warped = window_partition(shifted_x_warped, self.window_size)  # nw*b, window_size, window_size, c
            x_windows_warped = x_windows_warped.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c
        else:
            x_windows_warped = None

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, x_windows_warped, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, x_windows_warped, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(b, h * w, c)
        x_maconv = x_maconv.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x) + conv_x*self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x))) + x_maconv
        

        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        # norm1
        flops += self.dim * h * w
        # W-MSA/SW-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * h * w
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 vit_condition=0,
                 vit_condition_type='1conv',
                 condition_dim=1,
                 window_condition=False,
                 window_condition_only=False,
                 c_dim=60,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # self.add_maconv = MAConv(dim,dim,1, 1, 0, bias=True)

        use_vit_condition = [i >= depth - vit_condition for i in range(depth)]
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_vit_condition=use_vit_condition[i],
                condition_dim=condition_dim,
                vit_condition_type=vit_condition_type,
                window_condition=window_condition,
                window_condition_only=window_condition_only,
                c_dim=c_dim,
            ) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, condition): 
        # print(x.shape,'------------------------------------')   # [20, 4096,60]
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size, condition)
            
        # for i in range(self.depth):
        #     x1 = self.add_maconv(x, condition)
        #     x2 = self.blocks[i](x, x_size, condition)
        #     x = x1 + x2

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 vit_condition=0,
                 vit_condition_type='1conv',
                 condition_dim=1,
                 use_dcn_conv=False,
                 dcn_condition_type='1conv',
                 window_condition=False,
                 window_condition_only=False,
                 c_dim=60,
                 ):
        super(RSTB, self).__init__()

        # self.conv_sf1 = nn.Conv2d(dim, dim,3, 1,1)
        self.fam = FAM(dim, dim, 3,1,1, bias=True, split=2, reduction=2)
        # self.sam = SAM(dim, dim, 3,1,1, bias=True, split=2, reduction=2)
        # self.conv_sf2 = nn.Conv2d(dim, dim,3, 1,1)

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            vit_condition=vit_condition,
            vit_condition_type=vit_condition_type,
            condition_dim=condition_dim,
            window_condition=window_condition,
            window_condition_only=window_condition_only,
            c_dim=c_dim,
        )

        if resi_connection == '1conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1))
            # self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
        elif resi_connection == '0conv':
            self.conv = nn.Identity()

        self.use_dcn_conv = use_dcn_conv
        if self.use_dcn_conv:
            if resi_connection != '0conv':
                self.conv.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            if dcn_condition_type == '1conv':
                self.offset_conv = nn.Sequential(nn.Conv2d(condition_dim, dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif dcn_condition_type == '2conv':
                self.offset_conv = nn.Sequential(nn.Conv2d(condition_dim, c_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(c_dim, dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))

            self.dcn = DCNv2Pack(dim, dim, 3, padding=1)
            # self.add_maconv = MAConv(dim, dim, 1,1,0, bias=True)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        
        self.conv_change = nn.Conv2d(dim*3, dim, 3, 1, 1)
        
    # x 已经patch_embed
    def forward(self, x, x_size, condition):
        if self.use_dcn_conv:
            x_fam = self.patch_unembed(x,x_size)

            # sam and fam module
            # x_orgin = x_fam
            # x_fam = self.conv_sf1(x_fam)
            # x_fam = self.sam(x_fam)
            x_fam = self.fam(x_fam)
            # x_fam = self.conv_sf2(x_fam) + x_orgin

            _x = self.conv(self.patch_unembed(self.residual_group(x, x_size, condition), x_size))
            _x1 = self.patch_unembed(x,x_size)
            _x1 = self.dcn(_x1, self.offset_conv(condition))
            _x = torch.cat((_x, _x1), dim=1)
            _x = torch.cat((_x, x_fam), dim=1)
            _x = self.conv_change(_x)
            return self.patch_embed(self.dcn(_x , self.offset_conv(condition))) + x 
        else:
            return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, condition), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class OSRT(nn.Module):
    r"""
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
        condition_dim (int): The dimension of input conditions
        c_dim (int): The dimension of condition-related hidden layers
        vit_condition (list): whether to apply DAAB. Default: None
        dcn_condition (list): whether to apply DACB. Default: None
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 condition_dim=1,
                 vit_condition=None,
                 vit_condition_type='1conv',
                 dcn_condition=None,
                 dcn_condition_type='1conv',
                 window_condition=False,
                 window_condition_only=False,
                 c_dim=60,
                 **kwargs):
        super(OSRT, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans

        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # self.lte = LTE(imnet_spec=None, hidden_dim=256)

        

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        if dcn_condition is None:
            dcn_condition = [0 for _ in range(self.num_layers + 1)]
        if vit_condition is None:
            vit_condition = [0 for _ in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                vit_condition=vit_condition[i_layer],
                vit_condition_type=vit_condition_type,
                condition_dim=condition_dim,
                use_dcn_conv=bool(dcn_condition[i_layer]),
                dcn_condition_type=dcn_condition_type,
                window_condition=window_condition,
                window_condition_only=window_condition_only,
                c_dim=c_dim,
                )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
            # self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        elif resi_connection == '0conv':
            self.conv_after_body = nn.Identity()

        self.use_dcn_conv = bool(dcn_condition[-1])
        if self.use_dcn_conv:
            if resi_connection != '0conv':
                self.conv_after_body.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            if dcn_condition_type == '1conv':
                self.offset_conv = nn.Sequential(nn.Conv2d(condition_dim, embed_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif dcn_condition_type == '2conv':
                self.offset_conv = nn.Sequential(nn.Conv2d(condition_dim, c_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(c_dim, embed_dim, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))

            self.dcn = DCNv2Pack(embed_dim, embed_dim, 3, padding=1)
            

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat, input_resolution=(patches_resolution[0], patches_resolution[1]))
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, condition):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, condition)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, condition):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        self.feat_coord = make_coord(x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1).contiguous() \
            .unsqueeze(0).expand(x.shape[0], 2, * x.shape[-2:])
        
        

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            if self.use_dcn_conv:
                _x = self.conv_after_body(self.forward_features(x, condition))
                x = self.dcn(_x , self.offset_conv(condition)) + x
            else:
                x = self.conv_after_body(self.forward_features(x, condition)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)

            # x_lte = self.lte(x, self.feat_coord)
            # print(x_lte.shape,'111111111111111')

            if self.use_dcn_conv:
                _x = self.conv_after_body(self.forward_features(x, condition))
                x = self.dcn(_x , self.offset_conv(condition)) + x
            else:
                x = self.conv_after_body(self.forward_features(x, condition)) + x

            # print(x.shape,'222222222222222222222222')
            # x = torch.cat((x, x_lte),dim=1)
            

            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            if self.use_dcn_conv:
                _x = self.conv_after_body(self.forward_features(x, condition))
                x = self.dcn(_x , self.offset_conv(condition)) + x
            else:
                x = self.conv_after_body(self.forward_features(x, condition)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, condition)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    # full model
    model = OSRT(
        upscale=4,
        img_size=(64, 64),
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=156,
        c_dim=156,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
        condition_dim=1,
        vit_condition=[6, 6, 6, 6, 6, 6],
        vit_condition_type='3conv',
        dcn_condition=[1, 1, 1, 1, 1, 1, 1],
        dcn_condition_type='2conv',
        window_condition=True,
    )

    x = torch.randn((1, 3, 64, 64))
    c = torch.randn((1, 1, 64, 64))
    x = model(x, c)
    print(x.shape)
    print(sum(map(lambda x: x.numel(), model.parameters())))

    # light model
    model = OSRT(
        upscale=4,
        img_size=(64, 64),
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        c_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
        condition_dim=1,
        vit_condition=[6, 6, 6, 6],
        vit_condition_type='3conv',
        dcn_condition=[1, 1, 1, 1, 1],
        dcn_condition_type='2conv',
        window_condition=True,
    )

    x = torch.randn((1, 3, 64, 64))
    c = torch.randn((1, 1, 64, 64))
    x = model(x, c)
    print(x.shape)
    print(sum(map(lambda x: x.numel(), model.parameters())))
