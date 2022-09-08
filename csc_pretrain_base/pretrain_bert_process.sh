#!/usr/bin/env bash


name=pretrain

valid_data=./data/ccl_trck1_dev.txt
test_data=./data/13test.txt
bert_path=./plm_models/chinese_L-12_H-768_A-12/
# 预训练模型路径
train_data=./data/13train.txt
# 使用生成的数据路径替换
save=./save/pretrain_bert_csc/

if [ ! -d $save ]; then
  mkdir $save
fi

CUDA_VISIBLE_DEVICES="1" python bft_pretrain_mlm.py  --bert_path $bert_path --ignore_sep=False \
--task_name=$name --gpu_num=1 --load_model=False \
--do_train=True --train_data=$train_data \
--do_valid=True --valid_data=$valid_data --do_test=True --test_data=$test_data \
--epoch=10 --batch_size=64 --learning_rate=2e-5 --do_save=True \
--save_dir=$save --seed=10
# > $save/bft_train.log

