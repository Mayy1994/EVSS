#!/bin/bash
cd /home/mybeast/xjj/EVSS && \
python3 val_citys.py \
            --data_path /media/mybeast/beast \
            --gt_path /home/mybeast/xjj/cityscapes \
            --data_list_path /home/mybeast/xjj/EVSS/data/list/cityscapes/val.txt \
            --evss_model /home/mybeast/xjj/EVSS/saved_model/evss/evss_citys_best.pth \
            --num_workers 0 \