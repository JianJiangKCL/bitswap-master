
# python job_launcher.py -s full_ans.sh -gpu 1 --level bot --resume 1 -n full_bot
cd ..
LEVEL=$1
RESUME=$2
CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/IB_DRR_pl/results/saved_codes/nature_full_image/deflate_codes.npz

python code_train.py  --is_finetune ${RESUME}  --nz 2 --dataset img224 --dataset_path ${CODES_PATH} --code_level ${LEVEL} --width 128 --batch_size 128 --data_tag full