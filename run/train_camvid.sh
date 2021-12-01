#!/bin/bash
cd /home/mybeast/xjj/EVSS && \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train_camvid.py \
        --exp_name evss_camvid \
        --root_data_path /home/mybeast/xjj/CamVid_davss \
        --root_gt_path /home/mybeast/xjj/CamVid_davss \
        --train_list_path /home/mybeast/xjj/EVSS/data/list/camvid/trainval3.txt \
        --test_list_path /home/mybeast/xjj/EVSS/data/list/camvid/test3.txt \
        --lr 1e-4 \
        --conet_lr 1e-4 \
        --random_seed 666 \
        --weight_decay 0 \
        --train_batch_size 8 \
        --train_num_workers 8 \
        --test_batch_size 2 \
        --test_num_workers 4 \
        --num_epoch 100 \
        --snap_shot 5 \
        --model_save_path /home/mybeast/xjj/EVSS/evss_models/evss_camvid \
        --tblog_dir /home/mybeast/xjj/EVSS/tblog1/evss_camvid

