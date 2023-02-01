# bash run.sh
task="lift"
wandb_group="curl_0201"
seed=0
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name robot \
    --task_name lift \
    --action_space xyzw \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp \
    --agent curl_sac --frame_stack 3 \
    --seed ${seed} --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 5000 --batch_size 128 --num_train_steps 50000 \
    --use_wandb 1 \
    --wandb_project "robot_${task}_pretrain" \
    --wandb_group ${wandb_group} \
    --wandb_name ${seed}
