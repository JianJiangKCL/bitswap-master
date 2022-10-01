#!/bin/bash

cd ..

# np is not relevant for method itself, it is like the threads
CODES_PATH=/vol/jj/proj/bitswap/sub_down_img224_all_viaCifar.npz
python code_train.py  --is_finetune 0  --nz 1  --dataset img224 --dataset_path ${CODES_PATH} --code_level top --width 128 --batch_size 512 # --width 256 #--code_level bot #--zchannels 4 --kernel 1