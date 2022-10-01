#!/bin/bash
# set the number of nodes
# for small parti, if set nodes is 8, will cause PartiaionNodeLimit
#SBATCH --nodes=1
# set number of GPUs
#SBATCH --gres=gpu:4
# for big node, the max if 24
# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=ANS_topr

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jian.jiang@kcl.ac.uk

# run the application
cd ..
source ~/.bashrc
conda activate KM
# np is not relevant for method itself, it is like the threads
CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/bitswap/sub_down_img224_all_viaCifar.npz
python code_train_pl.py -c configs/bitswap_imagenet.yaml --resume 0    --dataset_path ${CODES_PATH} --code_level top  --gpus 4 --distributed --nz 1 --width 256  --tag sub --use_ema 1 -t lr=0.002  -t num_workers=8 -t batch_size=512 -t use_amp=0  # --width 256 #--code_level bot #--zchannels 4 --kernel 1