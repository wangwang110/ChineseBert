#!/usr/bin/env bash


train_data=/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_train.txt
valid_data=/data_local/TwoWaysToImproveCSC/BERT/data/rep_autog_wang_1k_dev.txt


save_path=/data_local/ChineseBert/save/wang2018/
CUDA_VISIBLE_DEVICES="6" python csc_train_mlm.py --task_name=test --gpu_num=1 --gradient_accumulation_steps=2 --load_model=False --do_train=True --train_data=$train_data --do_valid=True --valid_data=$valid_data --epoch=10 --batch_size=10 --learning_rate=2e-5 --do_save=True --save_dir=$save_path --seed=10 > $save_path/bft_train.log 2>&1 &