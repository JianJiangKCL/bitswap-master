# python job_launcher.py -s sub_ans.sh -gpu 1 --level top --resume 1 -n sub_top
cd ..
LEVEL=$1
RESUME=$2
#CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/IB_DRR_pl/results/saved_codes/nature_full_image/deflate_codes.npz
CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/bitswap/sub_down_img224_all_viaCifar.npz
# mpiexec -np 1 --oversubscribe python code_train.py --nz 4 --kernel 1 --code_level top --dataset cifar --dist 1 --gpu 4
# mpirun -np 4 python code_train.py --nz 4 --kernel 1 --code_level top --dataset cifar --gpu 4 --dist 1
python code_train.py  --is_finetune ${RESUME}  --nz 2 --dataset img224 --dataset_path ${CODES_PATH} --code_level ${LEVEL} --width 128 --batch_size 128