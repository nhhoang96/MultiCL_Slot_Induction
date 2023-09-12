conda init bash
source ~/.bashrc
conda activate classEnv

cd ../../../


python setup.py
pip install -e .
cd ./code/main_code/

ckpt='../ckpt/atis_release/' #Original train chkpt (old: reduce) #recent: score_ckpt_reward, score_reward_new #tmp_word (most recent word aug) -- tmp_word_complete (for prevnext) tmp_word_single_01_newmask
m='mask_seg'
seg_ratio=0.2
b=16
level=4
epoch=10
data='ATIS_P1'
lr=1e-5

seed=(123)
sent_temp=0.1
seg_temp=0.05
sent_coeff=0.2
seg_coeff=1.0

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
        --gpu_id=1 \
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
