#!/bin/bash
cd /home/mybeast/xjj/EVSS && \
python3 val_camvid.py \
            --data_path /home/mybeast/xjj/CamVid_davss \
            --gt_path /home/mybeast/xjj/CamVid_davss \
            --data_list_path /home/mybeast/xjj/EVSS/data/list/camvid/test3.txt \
            --evss_model /home/mybeast/xjj/EVSS/saved_model/evss/evss_camvid_best.pth \
            --num_workers 0 \