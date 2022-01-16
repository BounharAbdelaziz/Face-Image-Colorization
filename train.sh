export CUDA_VISIBLE_DEVICES=2
export CUDA_LAUNCH_BLOCKING=1


python3 -W ignore train.py --net_name 'cyclegan'\
                --batch_size 3 --n_epochs 200 --norm_type 'in2d' --experiment_name 'FACE_colo_512_cyclegan_bs_3_lr_1e_4_bicubic_in2d_stepLR_ffhq' \
                --print_freq 20 --lr 0.0001 --lr_policy 'step' --warmup_period 0 \
                --lambda_G 1 --lambda_D 1 --lambda_cycle 10 --lambda_L1 0 --lambda_MSE 0 --lambda_PCP 0 \
                --img_size 512 --img_dir '../datasets/ffhq_mini/images/' 
                #--verbose