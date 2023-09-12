conda init bash
source ~/.bashrc
conda activate si-multi
cd ../../../

python setup.py
pip install -e .
cd ./code/main_code/

ckpt='../ckpt/test_release/'
m='mask_seg'
seg_ratio=0.2
b=16
level=3
epoch=10
data='SNIPS_P1'
seed=(123)
lr=1e-5
seg_temp=0.1
sent_temp=0.05
seg_coeff=0.3
sent_coeff=0.7

for n in "${seed[@]}"; do
	python3 -u main_cl.py \
		--ckpt_dir=$ckpt \
		--num_run=$n \
		--dataset=$data \
		--seg_level $level \
		--batch_size=$b \
		--level_aug='seg' \
		--mask_type=$m \
		--learning_rate=$lr \
		--gpu_id=0 \
		--fixed_seg='false' \
		--ratio_seg=$seg_ratio \
		--num_epochs=$epoch \
		--write_log='false' \
		--self_supervise='true' \
		--augment='true' \
		--seg_temp=$seg_temp \
		--sent_temp=$sent_temp \
		--seg_coeff=$seg_coeff \
		--sent_coeff=$sent_coeff \
		--test_mode='false' \
		--cur_seed=$n
done

