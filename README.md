# RepSC-PA
# Environment
We conducted experiments using the same Linux environment as [OSRT](https://github.com/Fanghua-Yu).

# Data Preparation
The data download and preprocessing are consistent with the data preparation work in OSRT。

# Training
All experiments were conducted on an RTX 3090.
Training RepSC-PA:
```Bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=4 --master_port=7777 train.py -opt ./options/train/*.yml --launcher pytorch
```

#Testing
Testing RepSC-PA:
```Bash
CUDA_VISIBLE_DEVICES=0 python test.py -opt ./options/test/*.yml
```
