#!/bin/bash

cd ..

# np is not relevant for method itself, it is like the threads
CODES_PATH=/vol/jj/proj/bitswap/sub_down_img224_all_viaCifar.npz
python code_train_pl.py -c configs/bitswap_imagenet.yaml  --resume 0    --dataset_path ${CODES_PATH} --code_level top  -t num_workers=4 -t batch_size=256 -t gpus=1 --nz 1 --width 256 -t lr=0.01 --tag sub # --width 256 #--code_level bot #--zchannels 4 --kernel 1