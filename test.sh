export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1


python -W ignore test.py --net_name 'colorization'  --batch_size 1 \
                --norm_type 'in2d' --pretrained_path './check_points/FACE_colo_512_bs_6_lr_1e_4_bicubic_in2d_stepLR_ffhq/G_it_13602.pth' \
                --img_size 512 --img_dir './test_images/' --eval
                #--verbose