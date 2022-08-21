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
#SBATCH --job-name=cifar

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jian.jiang@kcl.ac.uk

# somehow cifar ddp is super slow; also single gpu is slow
# run the application
cd ..
source ~/.bashrc
conda activate KM
# np is not relevant for method itself, it is like the threads
CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/bitswap/cifar_uint16_via50VQVAEn512.npz
python code_train_pl.py -c configs/bitswap_cifar.yaml --resume 0    --dataset_path ${CODES_PATH} --code_level top  --gpus 1  -t num_workers=0 -t batch_size=1024 --nz 1 --width 128 #--distributed --width 256 #--code_level bot #--zchannels 4 --kernel 1