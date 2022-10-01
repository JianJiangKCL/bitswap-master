#!/bin/bash
# set the number of nodes
# for small partition, if set nodes is 8, will cause PartiaionNodeLimit
#SBATCH --nodes=1
# set number of GPUs
#SBATCH --gres=gpu:1
# for big node, the max if 24
# set max wallclock time
#SBATCH --time=144:00:00

# set name of job
#SBATCH --job-name=ANS_bot

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jian.jiang@kcl.ac.uk

# run the application
source /jmain01/home/JAD014/mxm09/jxj51-mxm09/envtest/bin/activate
python code_train.py --is_finetune 1  --nz 8 --kernel 1 --dataset img224 --code_level bot
