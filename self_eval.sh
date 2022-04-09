#!/usr/bin/env bash
# 使用论文公开的模型进行测试，分析


baseline="/data_local/ChineseBert/save/wang2018/model.pkl"
preTrain="/data_local/ChineseBert/save/wang2018/sighan13/model.pkl"


gpu=6


data=/data_local/TwoWaysToImproveCSC/BERT/data/13test_lower.txt
task=test
# new_pretrain_auto.dev
# sighan13
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu python csc_train_mlm.py  --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python csc_train_mlm.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data --batch_size=16


data=/data_local/TwoWaysToImproveCSC/BERT/cc_data/chinese_spell_lower_4.txt

## xaioxue
echo $baseline
CUDA_VISIBLE_DEVICES=$gpu  python csc_train_mlm.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$baseline --do_test=True --test_data=$data --batch_size=16

echo $preTrain
CUDA_VISIBLE_DEVICES=$gpu  python csc_train_mlm.py --task_name=$task --gpu_num=1 --load_model=True  --load_path=$preTrain --do_test=True --test_data=$data  --batch_size=16

