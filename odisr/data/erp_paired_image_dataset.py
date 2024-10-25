import math
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from .utils import paired_random_crop
import numpy as np
import cv2
import torch
import os.path as osp
from torch import nn

@DATASET_REGISTRY.register()
class ERPPairedImageDataset(data.Dataset):
    """Paired image dataset for conditional ODI restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ERPPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.pos_emb = PositionEmbeddingSine(4//2, normalize=True)

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if 'ext_dataroot_gt' in self.opt:
            assert self.io_backend_opt['type'] == 'disk'
            self.ext_gt_folder, self.ext_lq_folder = opt['ext_dataroot_gt'], opt['ext_dataroot_lq']
            if 'enlarge_scale' in self.opt:
                enlarge_scale = self.opt['enlarge_scale']
            else:
                enlarge_scale = [1 for _ in range(len(self.ext_gt_folder)+1)]

            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl) \
                         * enlarge_scale[0]
            for i in range(len(self.ext_gt_folder)):
                self.paths += paired_paths_from_folder([self.ext_lq_folder[i], self.ext_gt_folder[i]], ['lq', 'gt'],
                                                          self.filename_tmpl) * enlarge_scale[i+1]
        else:
            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
                self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                              self.opt['meta_info_file'], self.filename_tmpl)
            else:
                self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        if 'gt_size' in self.opt and self.opt['gt_size']:
            self.glob_condition = get_condition(self.opt['gt_h']//self.opt['scale'],
                                                self.opt['gt_w']//self.opt['scale'], self.opt['condition_type'])

        if 'sub_image' in self.opt and self.opt['sub_image']:
            self.sub_image = True
        else:
            self.sub_image = False


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        if self.sub_image:
            sub_h, sub_w = osp.split(lq_path)[-1].split('_')[3:5]
            sub_h, sub_w = int(sub_h) // scale, int(sub_w) // scale
        else:
            sub_h, sub_w = 0, 0

        if self.opt.get('force_resize'):
        # resize gt with wrong resolutions
            img_gt = cv2.resize(img_gt, (img_lq.shape[1] * scale, img_lq.shape[0] * scale), cv2.INTER_CUBIC)

        # augmentation for training
        # random crop
        if 'gt_size' in self.opt and self.opt['gt_size']:
            gt_size = self.opt['gt_size']
            img_gt, img_lq, top_lq, left_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path,
                                                                       return_top_left=True)
            top_lq, left_lq = top_lq + sub_h, left_lq + sub_w
            if self.opt['condition_type'] is not None:
                if ('DIV2K' or 'Flickr2K') in lq_path:
                    _condition = torch.zeros([1, img_lq.shape[0], img_lq.shape[1]])
                else:
                    _condition = self.glob_condition[:,top_lq:top_lq+img_lq.shape[0],left_lq:left_lq+img_lq.shape[1]]
            else:
                _condition = 0.
        else:
            _condition = get_condition(img_lq.shape[0], img_lq.shape[1], self.opt['condition_type'])
        if self.opt['phase'] == 'train':
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)


        # mask = torch.ones(1, img_lq.shape[1], img_lq.shape[2], dtype=torch.bool) 
        # # print(mask.shape, '222222222')  # [1, 64, 64]
        # _position = NestedTensor(tensors=img_lq.unsqueeze(dim=0), mask=mask)
        # # print(img_lq.unsqueeze(dim=0).shape, '33333333333')
        # # print(_position.shape, '44444444')
        # _position = self.pos_emb(_position)
        # # print(_position.shape, '111111111', _condition.shape) #[1, 4, 64, 64]  [1, 64, 64]
        # # print(_position.squeeze(0).shape)
        # #add or cat
        # # _condition = torch.cat((_condition , _position.squeeze(0)),dim=0)
        # _condition = _condition + _position.squeeze(0)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'condition': _condition}

    def __len__(self):
        return len(self.paths)



def get_condition(h, w, condition_type):
    if condition_type is None:
        return 0.
    elif condition_type == 'cos_latitude':
        return torch.cos(make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1).contiguous() * math.pi / 2)
    elif condition_type == 'latitude':
        return make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2, 0, 1).contiguous() * math.pi / 2
    elif condition_type == 'coord':
        return make_coord([h, w]).permute(2, 0, 1).contiguous()
    else:
        raise RuntimeError('Unsupported condition type')


def make_coord(shape, ranges=(-1, 1), flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask
        if mask == 'auto':
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError("tensors dim must be 3 or 4 but {}({})".format(self.tensors.dim(), self.tensors.shape))
 
    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res
 
    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
 
    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(tensor.dim())
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img
 
    def to_img_list(self):
        """remove the padding and convert to img list
        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res
 
    @property
    def device(self):
        return self.tensors.device
 
    def decompose(self):
        return self.tensors, self.mask
 
    def __repr__(self):
        return str(self.tensors)
 
    @property
    def shape(self):
        return {
            'tensors.shape': self.tensors.shape,
            'mask.shape': self.mask.shape
        }
 
#code is from DETR    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
 
    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        #沿h维
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32) #[bs,h,w]
        # print(y_embed.shape, '0000')
        #沿w维
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32) #[bs,h,w]
        # print(x_embed.shape, '1111')

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
 
        dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32, device=x.device) #[num_pos_feats]
        # print(dim_t.shape, '5555')
        #temperature就是原Transformer paper中的10000
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) #[num_pos_feats]
        # print(dim_t.shape, '6666')
 
        pos_x = x_embed[:, :, :, None] / dim_t  #[bs,h,w,dim/2]
        # print(pos_x.shape, '2222')
        pos_y = y_embed[:, :, :, None] / dim_t  #[bs,h,w,dim/2]
        # print(pos_y.shape, '3333')
        #可以看到pos_x和pos_y是一样的, 偶数维度sin，奇数维度cos
        #这里的做法就是偶数维度前一半，奇数维度后一半，然后拼起来
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  #[bs,h,w,dim/2]
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  #[bs,h,w,dim/2]
        #y维(h维)的position和x维(w维)的position 再拼起来
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  #[bs,dim,h,w]
        return pos