# from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from basicsr.archs.arch_util import to_2tuple, trunc_normal_, flow_warp, DCNv2Pack
from odisr.archs.esrt import Updownblock
# from odisr.archs.osrt_arch import SCPA

import torch
# from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from numpy import random


class Concat1x1and3x3(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
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
            return self.rbr_reparam(inputs)
        else:
            return self.rbr_3x3_branch(inputs) + self.rbr_1x1_branch(inputs)
        
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
    
    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (1, 1, 1, 1)) 

        bias = conv1.bias.data + conv2.bias.data 
        return weight, bias

                

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
            return self.rbr_reparam(inputs)
        else:
            return self.rbr_1x1_branch2(self.rbr_1x1_branch1(inputs)) 
        
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

class Connet3x3and3x3(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(Connet3x3and3x3, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=3, stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_3x3_branch2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')

    def forward(self, inputs):
        if (self.deploy):
            return self.rbr_reparam(inputs)
        else:
            return self.rbr_3x3_branch2(self.rbr_3x3_branch1(inputs)) 
        
    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                        stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # self.__delattr__('rbr_condition_branch')
        self.__delattr__('rbr_3x3_branch1')
        self.__delattr__('rbr_3x3_branch2')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
                # condition_branch
        # kernel_condition, bias_condition = self.rbr_condition_branch.weight.data, self.rbr_condition_branch.bias.data

        # 3x3 branch
        kernel_1 = self._fuse_3x3_3x3_branch(self.rbr_3x3_branch1,
                                                     self.rbr_3x3_branch2)

        # kernel_2, bias_2 = self.rbr_1x1_branch2.weight.data, self.rbr_1x1_branch2.bias.data

        return kernel_1 
    
    def _fuse_3x3_3x3_branch(self, conv1, conv2):
        weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
        return weight


class Rescpa(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(Rescpa, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        # self.group_width = self.in_channels // 2
        self.out_channels = out_channels
        self.activation = activation('lrelu')
        group_width = in_channels // 2
        if self.training:
            self.conv1 = Concat1x1and3x3(in_channels, group_width, act_type, deploy=deploy)
            self.conv2 = Concat1x1and3x3(in_channels, group_width, act_type, deploy=deploy)

            self.conv3 = Connet1x1and1x1(group_width, group_width, act_type, deploy=deploy)
            self.conv4 = Connet3x3and3x3(group_width, group_width, act_type, deploy=deploy)

            self.conv5 = Connet3x3and3x3(group_width, group_width, act_type, deploy=deploy)

            self.conv6 = Concat1x1and3x3(group_width, group_width, act_type, deploy=deploy)

            self.conv7 = Concat1x1and3x3(in_channels, out_channels, act_type, deploy=deploy)
        else:
            self.conv1 = Concat1x1and3x3(in_channels, group_width, act_type, deploy=deploy)
            self.conv2 = Concat1x1and3x3(in_channels, group_width, act_type, deploy=deploy)

            self.conv3 = Connet1x1and1x1(group_width, group_width, act_type, deploy=deploy)
            self.conv4 = Connet3x3and3x3(group_width, group_width, act_type, deploy=deploy)

            self.conv5 = Connet3x3and3x3(group_width, group_width, act_type, deploy=deploy)

            self.conv6 = Concat1x1and3x3(group_width, group_width, act_type, deploy=deploy)

            self.conv7 = Concat1x1and3x3(in_channels, out_channels, act_type, deploy=deploy)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # x1 = self.lrelu(self.conv1(x))
        # x2 = self.lrelu(self.conv2(x))
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        # x1 = self.lrelu(x1)
        # x2 = self.lrelu(x2)


        x11 = self.conv3(x1)
        x11 = self.sigmoid(x11)
        x12 = self.conv4(x1)
        x11 = torch.mul(x11, x12)
        x11 = self.conv6(x11)
        # x11 = self.lrelu(x11)

        # x11 = self.lrelu(self.conv6(x11))

        # x2 = self.lrelu(self.conv5(x2))
        x2 = self.conv5(x2)
        # x2 = self.lrelu(x2)

        x11 = torch.cat((x11, x2),dim=1)

        x11 = self.conv7(x11)
        x11 += x
        return x11 


class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        # self.offset_conv1 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
                                                #  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                #  nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
                                                #  nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # self.dcn1 = DCNv2Pack(nf, nf, 1, padding=0)


        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        
    def forward(self, x):

        # y = self.dcn1(x, self.offset_conv1(condition))
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        # self.offset_conv2 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                                          nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # self.dcn2 = DCNv2Pack(nf, group_width, 1, padding=0)

        # self.offset_conv3 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                                          nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # self.dcn3 = DCNv2Pack(nf, group_width, 1, padding=0)
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # self.fam = FAM(nf, nf, 3,1,1, bias=True, split=2, reduction=2)
        # self.conv_out = nn.Conv2d(nf*2, nf, 3, 1, 1)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        # out_a = self.dcn2(x, self.offset_conv2(condition))
        # out_b = self.dcn3(x, self.offset_conv3(condition))
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        # out = self.fam(out)

        out += residual
        # out = torch.cat((out, x_fam),dim=1)
        # out = self.conv_out(out)
        # out += x_fam
        return out

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class RepMLP(nn.Module):
    def __init__(self,C,O,H,W,h,w,fc1_fc2_reduction=1,fc3_groups=8,repconv_kernels=None,deploy=False):
        super().__init__()
        self.C=C
        self.O=O
        self.H=H
        self.W=W
        self.h=h
        self.w=w
        self.fc1_fc2_reduction=fc1_fc2_reduction
        self.repconv_kernels=repconv_kernels
        self.h_part=H//h
        self.w_part=W//w
        self.deploy=deploy
        self.fc3_groups=fc3_groups

        # make sure H,W can divided by h,w respectively
        assert H%h==0
        assert W%w==0

        self.is_global_perceptron= (H!=h) or (W!=w)
        ### global perceptron
        if(self.is_global_perceptron):
            if(not self.deploy):
                self.avg=nn.Sequential(OrderedDict([
                    ('avg',nn.AvgPool2d(kernel_size=(self.h,self.w))),
                    ('bn',nn.BatchNorm2d(num_features=C))
                ])
                )
            else:
                self.avg=nn.AvgPool2d(kernel_size=(self.h,self.w))
            hidden_dim=self.C//self.fc1_fc2_reduction
            self.fc1_fc2=nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(C*self.h_part*self.w_part,hidden_dim)),
                ('relu',nn.ReLU()),
                ('fc2',nn.Linear(hidden_dim,C*self.h_part*self.w_part))
            ])
            )

        # self.fc3=nn.Conv2d(self.C*self.h*self.w,self.O*self.h*self.w,kernel_size=1,groups=fc3_groups,bias=self.deploy)
        # self.fc3_bn=nn.Identity() if self.deploy else nn.BatchNorm2d(self.O*self.h*self.w)
        
        # if not self.deploy and self.repconv_kernels is not None:
        #     for k in self.repconv_kernels:
        #         repconv=nn.Sequential(OrderedDict([
        #             ('conv',nn.Conv2d(self.C,self.O,kernel_size=k,padding=(k-1)//2, groups=fc3_groups,bias=False)),
        #             ('bn',nn.BatchNorm2d(self.O))
        #         ])

        #         )
        #         self.__setattr__('repconv{}'.format(k),repconv)
                

    def switch_to_deploy(self):
        self.deploy=True
        # fc1_weight,fc1_bias,fc3_weight,fc3_bias=self.get_equivalent_fc1_fc3_params()
        fc1_weight,fc1_bias=self.get_equivalent_fc1_fc3_params()
        #del conv
        # if(self.repconv_kernels is not None):
        #     for k in self.repconv_kernels:
        #         self.__delattr__('repconv{}'.format(k))
        #del fc3,bn
        # self.__delattr__('fc3')
        # self.__delattr__('fc3_bn')
        # self.fc3 = nn.Conv2d(self.C * self.h * self.w, self.O * self.h * self.w, 1, 1, 0, bias=True, groups=self.fc3_groups)
        # self.fc3_bn = nn.Identity()
        #   Remove the BN after AVG
        if self.is_global_perceptron:
            self.__delattr__('avg')
            self.avg = nn.AvgPool2d(kernel_size=(self.h, self.w))
        #   Set values
        if fc1_weight is not None:
            self.fc1_fc2.fc1.weight.data = fc1_weight
            self.fc1_fc2.fc1.bias.data = fc1_bias
        # self.fc3.weight.data = fc3_weight
        # self.fc3.bias.data = fc3_bias




    def get_equivalent_fc1_fc3_params(self):
        #training fc3+bn weight
        # fc_weight,fc_bias=self._fuse_bn(self.fc3,self.fc3_bn)
        #training conv weight
        # if(self.repconv_kernels is not None):
        #     max_kernel=max(self.repconv_kernels)
        #     max_branch=self.__getattr__('repconv{}'.format(max_kernel))
        #     conv_weight,conv_bias=self._fuse_bn(max_branch.conv,max_branch.bn)
        #     for k in self.repconv_kernels:
        #         if(k!=max_kernel):
        #             tmp_branch=self.__getattr__('repconv{}'.format(k))
        #             tmp_weight,tmp_bias=self._fuse_bn(tmp_branch.conv,tmp_branch.bn)
        #             tmp_weight=F.pad(tmp_weight,[(max_kernel-k)//2]*4)
        #             conv_weight+=tmp_weight
        #             conv_bias+=tmp_bias
        #     repconv_weight,repconv_bias=self._conv_to_fc(conv_weight,conv_bias)
        #     final_fc3_weight=fc_weight+repconv_weight.reshape_as(fc_weight)
        #     final_fc3_bias=fc_bias+repconv_bias
        # else:
        #     final_fc3_weight=fc_weight
        #     final_fc3_bias=fc_bias

        #fc1
        if(self.is_global_perceptron):
            #remove BN after avg
            avgbn = self.avg.bn
            std = (avgbn.running_var + avgbn.eps).sqrt()
            scale = avgbn.weight / std
            avgbias = avgbn.bias - avgbn.running_mean * scale
            fc1 = self.fc1_fc2.fc1
            replicate_times = fc1.in_features // len(avgbias)
            replicated_avgbias = avgbias.repeat_interleave(replicate_times).view(-1, 1)
            bias_diff = fc1.weight.matmul(replicated_avgbias).squeeze()
            final_fc1_bias = fc1.bias + bias_diff
            final_fc1_weight = fc1.weight * scale.repeat_interleave(replicate_times).view(1, -1)

        else:
            final_fc1_weight=None
            final_fc1_bias=None
        
        return final_fc1_weight,final_fc1_bias
    # return final_fc1_weight,final_fc1_bias,final_fc3_weight,final_fc3_bias




    # def _conv_to_fc(self,weight,bias):
    #     i_maxtrix=torch.eye(self.C*self.h*self.w//self.fc3_groups).repeat(1,self.fc3_groups).reshape(self.C*self.h*self.w//self.fc3_groups,self.C,self.h,self.w)
    #     fc_weight=F.conv2d(i_maxtrix,weight=weight,bias=bias,padding=weight.shape[2]//2,groups=self.fc3_groups)
    #     fc_weight=fc_weight.reshape(self.C*self.h*self.w//self.fc3_groups,-1)
    #     fc_bias = bias.repeat_interleave(self.h * self.w)
    #     return fc_weight,fc_bias


    def _conv_to_fc(self,conv_kernel, conv_bias):
        I = torch.eye(self.C * self.h * self.w // self.fc3_groups).repeat(1, self.fc3_groups).reshape(self.C * self.h * self.w // self.fc3_groups, self.C, self.h, self.w).to(conv_kernel.device) 
        fc_k = F.conv2d(I, conv_kernel, padding=conv_kernel.size(2)//2, groups=self.fc3_groups)
        fc_k = fc_k.reshape(self.C * self.h * self.w // self.fc3_groups, self.O * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


    def _fuse_bn(self, conv_or_fc, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        if conv_or_fc.weight.ndim == 4:
            t = t.reshape(-1, 1, 1, 1)
        else:
            t = t.reshape(-1, 1)
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std


    def forward(self,x) :
        ### global partition
        if(self.is_global_perceptron):
            input=x
            v=self.avg(x) #bs,C,h_part,w_part
            v=v.reshape(-1,self.C*self.h_part*self.w_part) #bs,C*h_part*w_part
            v=self.fc1_fc2(v) #bs,C*h_part*w_part
            v=v.reshape(-1,self.C,self.h_part,1,self.w_part,1) #bs,C,h_part,w_part
            input=input.reshape(-1,self.C,self.h_part,self.h,self.w_part,self.w) #bs,C,h_part,h,w_part,w
            input=v+input
        else:
            input=x.view(-1,self.C,self.h_part,self.h,self.w_part,self.w) #bs,C,h_part,h,w_part,w
        # partition=input.permute(0,2,4,1,3,5) #bs,h_part,w_part,C,h,w

        partition = input.reshape(-1, self.C, self.H, self.W)
        ### partition partition
        # fc3_out=partition.reshape(-1,self.C*self.h*self.w,1,1) #bs*h_part*w_part,C*h*w,1,1
        # fc3_out=self.fc3_bn(self.fc3(fc3_out)) #bs*h_part*w_part,O*h*w,1,1
        # fc3_out=fc3_out.reshape(-1,self.h_part,self.w_part,self.O,self.h,self.w) #bs,h_part,w_part,O,h,w

        ### local perceptron
        # if(self.repconv_kernels is not None and not self.deploy):
        #     conv_input=partition.reshape(-1,self.C,self.h,self.w) #bs*h_part*w_part,C,h,w
        #     conv_out=0
        #     for k in self.repconv_kernels:
        #         repconv=self.__getattr__('repconv{}'.format(k))
        #         conv_out+=repconv(conv_input) ##bs*h_part*w_part,O,h,w
        #     conv_out=conv_out.view(-1,self.h_part,self.w_part,self.O,self.h,self.w) #bs,h_part,w_part,O,h,w
        #     fc3_out+=conv_out
        # fc3_out=fc3_out.permute(0,3,1,4,2,5)#bs,O,h_part,h,w_part,w
        # fc3_out=fc3_out.reshape(-1,self.C,self.H,self.W) #bs,O,H,W


        return partition



# if __name__ == '__main__':
#     setup_seed(20)
#     N=4 #batch size
#     C=512 #input dim
#     O=1024 #output dim
#     H=14 #image height
#     W=14 #image width
#     h=7 #patch height
#     w=7 #patch width
#     fc1_fc2_reduction=1 #reduction ratio
#     fc3_groups=8 # groups
#     repconv_kernels=[1,3,5,7] #kernel list
#     repmlp=RepMLP(C,O,H,W,h,w,fc1_fc2_reduction,fc3_groups,repconv_kernels=repconv_kernels)
#     x=torch.randn(N,C,H,W)
#     repmlp.eval()
#     for module in repmlp.modules():
#         if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
#             nn.init.uniform_(module.running_mean, 0, 0.1)
#             nn.init.uniform_(module.running_var, 0, 0.1)
#             nn.init.uniform_(module.weight, 0, 0.1)
#             nn.init.uniform_(module.bias, 0, 0.1)

#     #training result
#     out=repmlp(x)


#     #inference result
#     repmlp.switch_to_deploy()
#     deployout = repmlp(x)

#     print(((deployout-out)**2).sum())



def conv_layer(in_channels, out_channels, kernel_size, stride=1):
    # kernel_size参数预处理
    if not isinstance(kernel_size, collections.abc.Iterable):
        kernel_size = tuple(repeat(kernel_size, 2))
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


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


# thanks for ECBSR: https://github.com/xindongzhang/ECBSR
class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        if self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        tmp = self.scale * self.mask
        k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
        for i in range(self.out_planes):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        b1 = self.bias
        # re-param conv kernel
        RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
        # re-param conv bias
        RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB

# class RepBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, act_type, deploy=False):
#         super(RepBlock, self).__init__()
#         self.deploy = deploy
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.activation = activation('lrelu')

#         if deploy:
#             self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                          kernel_size=(3, 3), stride=1,
#                                          padding=1, dilation=1, groups=1, bias=True,
#                                          padding_mode='zeros')
#         else:
#             self.rbr_condition_branch = DCNv2Pack(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
#                                             stride=1, padding=1)
#             self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
#                                             stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
#             self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
#                                             stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
#             self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
#                                             stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
#             self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
#                                             stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')
#             self.rbr_1x1_3x3_branch_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels,
#                                                     kernel_size=(1, 1),
#                                                     stride=1, padding=(0, 0), dilation=1, groups=1,
#                                                     padding_mode='zeros', bias=False)
#             self.rbr_1x1_3x3_branch_3x3 = nn.Conv2d(in_channels=2 * in_channels, out_channels=out_channels,
#                                                     kernel_size=(3, 3),
#                                                     stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros',
#                                                     bias=False)
#             self.rbr_conv1x1_sbx_branch = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
#             self.rbr_conv1x1_sby_branch = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
#             self.rbr_conv1x1_lpl_branch = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)

#     def forward(self, inputs, offset):
#         if (self.deploy):
#             return self.activation(self.rbr_reparam(inputs))
#         else:
#             return self.activation(
#                 self.rbr_condition_branch(inputs, offset) + self.rbr_3x3_branch(inputs) + self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(
#                     inputs) + self.rbr_1x1_branch(inputs) + self.rbr_1x1_3x3_branch_3x3(
#                     self.rbr_1x1_3x3_branch_1x1(inputs)) + inputs + self.rbr_conv1x1_sbx_branch(
#                     inputs) + self.rbr_conv1x1_sby_branch(inputs) + self.rbr_conv1x1_lpl_branch(inputs))

#     def switch_to_deploy(self):
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
#                                      stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
#         self.rbr_reparam.weight.data = kernel
#         self.rbr_reparam.bias.data = bias
#         self.__delattr__('rbr_condition_branch')
#         self.__delattr__('rbr_3x3_branch')
#         self.__delattr__('rbr_3x1_branch')
#         self.__delattr__('rbr_1x3_branch')
#         self.__delattr__('rbr_1x1_branch')
#         self.__delattr__('rbr_1x1_3x3_branch_1x1')
#         self.__delattr__('rbr_1x1_3x3_branch_3x3')
#         self.__delattr__('rbr_conv1x1_sbx_branch')
#         self.__delattr__('rbr_conv1x1_sby_branch')
#         self.__delattr__('rbr_conv1x1_lpl_branch')
#         self.deploy = True

#     def get_equivalent_kernel_bias(self):
#                 # condition_branch
#         kernel_condition, bias_condition = self.rbr_condition_branch.weight.data, self.rbr_condition_branch.bias.data

#         # 3x3 branch
#         kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data

#         # 1x1 1x3 3x1 branch
#         kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
#                                                                                        self.rbr_1x3_branch,
#                                                                                        self.rbr_3x1_branch)
#         # 1x1+3x3 branch
#         kernel_1x1_3x3_fuse = self._fuse_1x1_3x3_branch(self.rbr_1x1_3x3_branch_1x1,
#                                                         self.rbr_1x1_3x3_branch_3x3)
#         # identity branch
#         device = kernel_1x1_3x3_fuse.device  # just for getting the device
#         kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
#         for i in range(self.out_channels):
#             kernel_identity[i, i, 1, 1] = 1.0

#         kernel_1x1_sbx, bias_1x1_sbx = self.rbr_conv1x1_sbx_branch.rep_params()
#         kernel_1x1_sby, bias_1x1_sby = self.rbr_conv1x1_sby_branch.rep_params()
#         kernel_1x1_lpl, bias_1x1_lpl = self.rbr_conv1x1_lpl_branch.rep_params()

#         return kernel_condition+kernel_3x3 + kernel_1x1_1x3_3x1_fuse + kernel_1x1_3x3_fuse + kernel_identity + kernel_1x1_sbx + kernel_1x1_sby + kernel_1x1_lpl, bias_condition+bias_3x3 + bias_1x1_1x3_3x1_fuse + bias_1x1_sbx + bias_1x1_sby + bias_1x1_lpl

#     def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
#         weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
#             conv3.weight.data, (1, 1, 0, 0))
#         bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
#         return weight, bias

#     def _fuse_1x1_3x3_branch(self, conv1, conv2):
#         weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
#         return weight

class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            # self.rbr_condition_branch = DCNv2Pack(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
            #                                 stride=1, padding=1)
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            # self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
            #                                 stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
            # self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
            #                                 stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            # self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            # stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_3x3_branch_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels,
                                                    kernel_size=(1, 1),
                                                    stride=1, padding=(0, 0), dilation=1, groups=1,
                                                    padding_mode='zeros', bias=False)
            self.rbr_1x1_3x3_branch_3x3 = nn.Conv2d(in_channels=2 * in_channels, out_channels=out_channels,
                                                    kernel_size=(3, 3),
                                                    stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros',
                                                    bias=False)
            self.rbr_conv1x1_sbx_branch = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
            self.rbr_conv1x1_sby_branch = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
            self.rbr_conv1x1_lpl_branch = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)

    def forward(self, inputs):
        if (self.deploy):
            return self.activation(self.rbr_reparam(inputs))
        else:
            # return self.activation(
            #     self.rbr_3x3_branch(inputs) + self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(
            #         inputs) + self.rbr_1x1_branch(inputs) + self.rbr_1x1_3x3_branch_3x3(
            #         self.rbr_1x1_3x3_branch_1x1(inputs)) + inputs )
            return self.activation(
                    self.rbr_3x3_branch(inputs) + self.rbr_1x1_3x3_branch_3x3(
                        self.rbr_1x1_3x3_branch_1x1(inputs)) + inputs + self.rbr_conv1x1_sbx_branch(
                        inputs) + self.rbr_conv1x1_sby_branch(inputs) + self.rbr_conv1x1_lpl_branch(inputs))

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # self.__delattr__('rbr_condition_branch')
        self.__delattr__('rbr_3x3_branch')
        # self.__delattr__('rbr_3x1_branch')
        # self.__delattr__('rbr_1x3_branch')
        # self.__delattr__('rbr_1x1_branch')
        self.__delattr__('rbr_1x1_3x3_branch_1x1')
        self.__delattr__('rbr_1x1_3x3_branch_3x3')
        self.__delattr__('rbr_conv1x1_sbx_branch')
        self.__delattr__('rbr_conv1x1_sby_branch')
        self.__delattr__('rbr_conv1x1_lpl_branch')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
                # condition_branch
        # kernel_condition, bias_condition = self.rbr_condition_branch.weight.data, self.rbr_condition_branch.bias.data

        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data

        # 1x1 1x3 3x1 branch
        # kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
        #                                                                                self.rbr_1x3_branch,
        #                                                                                self.rbr_3x1_branch)
        # 1x1+3x3 branch
        kernel_1x1_3x3_fuse = self._fuse_1x1_3x3_branch(self.rbr_1x1_3x3_branch_1x1,
                                                        self.rbr_1x1_3x3_branch_3x3)
        # identity branch
        device = kernel_1x1_3x3_fuse.device  # just for getting the device
        kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            kernel_identity[i, i, 1, 1] = 1.0

        kernel_1x1_sbx, bias_1x1_sbx = self.rbr_conv1x1_sbx_branch.rep_params()
        kernel_1x1_sby, bias_1x1_sby = self.rbr_conv1x1_sby_branch.rep_params()
        kernel_1x1_lpl, bias_1x1_lpl = self.rbr_conv1x1_lpl_branch.rep_params()

        # return kernel_3x3 + kernel_1x1_1x3_3x1_fuse + kernel_1x1_3x3_fuse + kernel_identity , bias_3x3 + bias_1x1_1x3_3x1_fuse
        return kernel_3x3 + kernel_1x1_3x3_fuse + kernel_identity + kernel_1x1_sbx + kernel_1x1_sby + kernel_1x1_lpl, bias_3x3 + bias_1x1_sbx + bias_1x1_sby + bias_1x1_lpl
   
    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
            conv3.weight.data, (1, 1, 0, 0))
        bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
        return weight, bias

    def _fuse_1x1_3x3_branch(self, conv1, conv2):
        weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
        return weight


# 重参数
# class RepRFB(nn.Module):
#     def __init__(self, feature_nums, act_type='lrelu', deploy=False):
#         super(RepRFB, self).__init__()
#         self.repblock1 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
#         self.repblock2 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
#         self.repblock3 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)

#         self.conv3 = conv_layer(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3)

#         self.esa = ESA(16, feature_nums, nn.Conv2d)
#         self.act = activation('lrelu')

#     def forward(self, inputs, offset):
#         outputs = self.repblock1(inputs, offset)
#         outputs = self.repblock2(outputs, offset)
#         outputs = self.repblock3(outputs, offset)
#         outputs = self.act(self.conv3(outputs))
#         outputs = inputs + outputs

#         outputs = self.esa(outputs)
#         return outputs
    
class RepRFB(nn.Module):
    def __init__(self, feature_nums, act_type='lrelu', deploy=False):
        super(RepRFB, self).__init__()
        self.repblock1 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
        self.repblock2 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
        self.repblock3 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
        # self.repblock4 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
        self.conv3 = conv_layer(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3)

        self.esa = ESA(16, feature_nums, nn.Conv2d)
        self.act = activation('lrelu')

    def forward(self, inputs):
        outputs = self.repblock1(inputs)
        outputs = self.repblock2(outputs)
        outputs = self.repblock3(outputs)
        # outputs = self.repblock4(outputs)
        outputs = self.act(self.conv3(outputs))
        outputs = inputs + outputs

        outputs = self.esa(outputs)
        return outputs



class Upsample_Block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
        super(Upsample_Block, self).__init__()
        self.conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        c = self.conv(input)
        upsample = self.pixel_shuffle(c)
        return upsample


# thanks for RLFN: https://github.com/bytedance/RLFN
class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        #         self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        #         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        #         v_range = self.relu(self.conv_max(v_max))
        #         c3 = self.relu(self.conv3(v_range))
        #         c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


# class RepRFN(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, feature_nums=48, upscale_factor=4,
#                  deploy=False):
#         super(RepRFN, self).__init__()
#         # self.fea_conv = conv_layer(in_channels=in_channels, out_channels=feature_nums, kernel_size=3)

#         self.reprfb1 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
#         self.reprfb2 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
#         self.reprfb3 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
#         self.reprfb4 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)

#         self.lr_conv = conv_layer(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3)

#         # self.upsampler = Upsample_Block(in_channels=feature_nums, out_channels=out_channels,
#         #                                 upscale_factor=upscale_factor)

#     def forward(self, inputs,offset):
#         # outputs_feature = self.fea_conv(inputs)

#         outputs = self.reprfb1(inputs, offset)
#         outputs = self.reprfb2(outputs, offset)
#         outputs = self.reprfb3(outputs, offset)
#         outputs = self.reprfb4(outputs, offset)
#         outputs = self.lr_conv(outputs)

#         outputs = outputs + inputs

#         # outputs = self.upsampler(outputs)
#         return outputs


class RepRFN(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, feature_nums=48, upscale_factor=4,
                 deploy=False):
        super(RepRFN, self).__init__()
        # self.H =H 
        # self.W = W 
        # self.fea_conv = conv_layer(in_channels=in_channels, out_channels=feature_nums, kernel_size=3)
        # self.updown1= Updownblock(n_feats=feature_nums)
        # self.scpa1 = SCPA(60,2,1,1)
        # self.glob1 = RepMLP(C=60,O=60,H=self.H,W=self.W,h=8,w=8,fc1_fc2_reduction=1,fc3_groups=8,repconv_kernels=None,deploy=deploy)
        # self.fam1 = FAM(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3, stride=1, padding=1, bias=True, split=2, reduction=2)
        self.reprfb1 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
        # self.updown2= Updownblock(n_feats=feature_nums)
        # self.scpa2 = SCPA(60,2,1,1)
        # self.glob2 = RepMLP(C=60,O=60,H=self.H,W=self.W,h=8,w=8,fc1_fc2_reduction=1,fc3_groups=8,repconv_kernels=None,deploy=deploy)
        # self.fam2 = FAM(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3, stride=1, padding=1, bias=True, split=2, reduction=2)
        self.reprfb2 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
        # self.updown3= Updownblock(n_feats=feature_nums)
        # self.scpa3 = SCPA(60,2,1,1)
        # self.glob3 = RepMLP(C=60,O=60,H=self.H,W=self.W,h=8,w=8,fc1_fc2_reduction=1,fc3_groups=8,repconv_kernels=None,deploy=deploy)
        # self.fam3 = FAM(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3, stride=1, padding=1, bias=True, split=2, reduction=2)
        self.reprfb3 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
        # self.updown4= Updownblock(n_feats=feature_nums)
        # self.scpa4 = SCPA(60,2,1,1)
        # self.glob4 = RepMLP(C=60,O=60,H=self.H,W=self.W,h=8,w=8,fc1_fc2_reduction=1,fc3_groups=8,repconv_kernels=None,deploy=deploy)
        # self.fam4 = FAM(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3, stride=1, padding=1, bias=True, split=2, reduction=2)
        self.reprfb4 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
        # self.reprfb5 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)

        self.lr_conv = conv_layer(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3)

        # self.upsampler = Upsample_Block(in_channels=feature_nums, out_channels=out_channels,
        #                                 upscale_factor=upscale_factor)

    def forward(self, inputs):
        # outputs_feature = self.fea_conv(inputs)

        outputs = self.reprfb1(inputs)
        # outputs = self.scpa1(outputs)
        # outputs = self.glob1(outputs)
        outputs = self.reprfb2(outputs)
        # outputs = self.scpa2(outputs)
        # outputs = self.glob2(outputs)
        outputs = self.reprfb3(outputs)
        # outputs = self.scpa3(outputs)
        # outputs = self.glob3(outputs)
        outputs = self.reprfb4(outputs)
        # outputs = self.scpa4(outputs)
        # outputs = self.glob4(outputs)
        # outputs = self.reprfb5(outputs)
        outputs = self.lr_conv(outputs)

        outputs = outputs + inputs

        # outputs = self.upsampler(outputs)
        return outputs


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
        # x_g = x_g * self.para2
        # x_g_sub = x_g_sub * self.para1
        # x_a = x_a * self.para4
        # x_a_sub = x_a_sub * self.para3

        return x_g + x_g_sub + x_a + x_a_sub
    
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
    
class SCPA_rep(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1, act_type='lrelu', deploy=False):
        super(SCPA_rep, self).__init__()
        group_width = nf // reduction

        self.repblock1 = RepBlock(in_channels=nf, out_channels=nf, act_type=act_type, deploy=deploy)
        # self.repblock2 = RepBlock(in_channels=nf, out_channels=group_width, act_type=act_type, deploy=deploy)
        # self.offset_conv2 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                                          nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # self.dcn2 = DCNv2Pack(nf, group_width, 1, padding=0)

        # self.offset_conv3 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                                          nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
        #                                          nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # self.dcn3 = DCNv2Pack(nf, group_width, 1, padding=0)
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fam = FAM(nf, nf, 3,1,1, bias=True, split=2, reduction=2)
        # self.conv_out = nn.Conv2d(nf*2, nf, 3, 1, 1)
        self.conv_out = BSConvU(nf*2, nf)

    def forward(self, x):
        residual = x


        repx = self.repblock1(x)
        out_a= self.conv1_a(repx)
        out_b = self.conv1_b(repx)
        # out_b = self.repblock2(x)
        # out_a = self.dcn2(x, self.offset_conv2(condition))
        # out_b = self.dcn3(x, self.offset_conv3(condition))
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        x_fam = self.fam(out)

        # out += residual
        out = torch.cat((out, x_fam),dim=1)
        out = self.conv_out(out)
        out += residual
        return out

class PAConv_offset(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv_offset, self).__init__()
        # self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.offset_conv1 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.dcn1 = DCNv2Pack(nf, nf, 1, padding=0)


        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        
    def forward(self, x, condition):

        y = self.dcn1(x, self.offset_conv1(condition))
        # y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class SCPA_rep_offset(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1, act_type='lrelu', deploy=False):
        super(SCPA_rep_offset, self).__init__()
        group_width = nf // reduction

        self.repblock1 = RepBlock(in_channels=nf, out_channels=nf, act_type=act_type, deploy=deploy)
        # self.repblock2 = RepBlock(in_channels=nf, out_channels=group_width, act_type=act_type, deploy=deploy)
        self.offset_conv2 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.dcn2 = DCNv2Pack(nf, group_width, 1, padding=0)

        self.offset_conv3 = nn.Sequential(nn.Conv2d(1, nf*2, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.dcn3 = DCNv2Pack(nf, group_width, 1, padding=0)
        
        # self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        # self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv_offset(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.fam = FAM(nf, nf, 3,1,1, bias=True, split=2, reduction=2)
        # self.conv_out = nn.Conv2d(nf*2, nf, 3, 1, 1)
        self.conv_out = BSConvU(nf*2, nf)

    def forward(self, x, condition):
        residual = x


        repx = self.repblock1(x)
        # out_a= self.conv1_a(repx)
        # out_b = self.conv1_b(repx)
        # out_b = self.repblock2(x)
        out_a = self.dcn2(repx, self.offset_conv2(condition))
        out_b = self.dcn3(repx, self.offset_conv3(condition))
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        x_fam = self.fam(out)

        # out += residual
        out = torch.cat((out, x_fam),dim=1)
        out = self.conv_out(out)
        out += residual
        return out

# if __name__ == '__main__':
#     from utils.model_summary import get_model_flops, get_model_activation

#     model = RepRFN(deploy=True)

#     input_dim = (3, 256, 256)  # set the input dimension
#     activations, num_conv = get_model_activation(model, input_dim)
#     activations = activations / 10 ** 6
#     print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
#     print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

#     flops = get_model_flops(model, input_dim, False)
#     flops = flops / 10 ** 9
#     print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

#     num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
#     num_parameters = num_parameters / 10 ** 6
#     print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
