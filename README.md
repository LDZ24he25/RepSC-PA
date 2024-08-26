##RepSC-PA


##Environment

Our environment is built based on [OSRT](https://github.com/Fanghua-Yu/OSRT)

##Data Preparation
The preprocessing of data is consistent with the operations in OSRT.

##Training
We use an RTX3090 for training.

Training RepSC-PA:
```Bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=4 --master_port=7777 train.py -opt ./options/train/*.yml --launcher pytorch
```

##Testing
Testing RepSC-PA:
```Bash
CUDA_VISIBLE_DEVICES=0 python test.py -opt ./options/test/*.yml
```
