#!/bin/bash
# set the number of nodes
# for small parti, if set nodes is 8, will cause PartiaionNodeLimit
#SBATCH --nodes=1
# set number of GPUs
#SBATCH --gres=gpu:1
# for big node, the max if 24
# set max wallclock time
#SBATCH --time=120:00:00

# set name of job
#SBATCH --job-name=ANS_bot

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jian.jiang@kcl.ac.uk

# run the application
cd ..
source ~/.bashrc
conda activate bitswap
# np is not relevant for method itself, it is like the threads
#CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/IB_DRR_pl/results/saved_codes/nature_full_image/deflate_codes.npz
CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/IB_DRR_pl/results/saved_codes/nature_full_image_500samples/deflate_codes.npz
python code_train.py --is_finetune 0  --nz 4  --dataset img224 --dataset_path ${CODES_PATH} --code_level bot --width 254 #--code_level bot #--zchannels 4 --kernel 1