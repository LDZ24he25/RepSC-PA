from odisr.archs import common

import torch.nn as nn
import torch

# print('this is edsr shift')


def make_model(args, parent=False):
    return EDSR(args)


def report_model(model):
    n_parameters = 0
    for p in model.parameters():
        n_parameters += p.nelement()
    print("Model  with {} parameters".format(n_parameters))


class EDSR(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 16
        n_feats = 60
        kernel_size = 3
        scale = 2
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(255)
        self.add_mean = common.MeanShift(255, sign=1)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock_shift(
                conv, n_feats, kernel_size, act=act, res_scale=10, move_pixel=2,
                move_channel=0
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
