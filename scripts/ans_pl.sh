

cd ..
LEVEL=$1
RESUME=$2
# np is not relevant for method itself, it is like the threads
CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/IB_DRR_pl/results/saved_codes/nature_full_image/deflate_codes.npz
#CODES_PATH=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/IB_DRR_pl/results/saved_codes/nature_full_image_500samples/deflate_codes.npz
python code_train_pl.py -c configs/bitswap_imagenet.yaml --resume ${RESUME}    --dataset_path ${CODES_PATH} --code_level ${LEVEL}  --gpus 4 --distributed -t num_workers=16 -t batch_size=1024 --nz 1 --width 128 # --width 256 #--code_level bot #--zchannels 4 --kernel 1