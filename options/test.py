import os.path as osp
import BasicSR.basicsr as basicsr
import odisr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    basicsr.test_pipeline(root_path)
