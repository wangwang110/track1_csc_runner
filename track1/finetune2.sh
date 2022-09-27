
# 第二阶段的微调

PRETRAIN_MODEL=/data_local/plm_models/chinese-roberta-wwm-ext/


step="2"
CUDA_DEVICE=5


tag="hsk"
DATA_BASE=./track1_data/train/
TRAIN_FILE=$DATA_BASE/hsk/hsk_error_word_one_uniq
# 使用hsk+sighan数据


#tag="all"
#DATA_BASE=./track1_data/train/
#TRAIN_FILE=$DATA_BASE/merge_one_wke
## 使用所有的拼写纠错数据


################# 不修改 #################

DATA_DIR="./data/"$tag
mkdir -p $DATA_DIR


TRAIN_SRC_FILE=$DATA_DIR/train.src
TRAIN_TRG_FILE=$DATA_DIR/train.trg


sigahn_src=./track1_data/sighan.train.ccl22.src
sigahn_trg=./track1_data/sighan.train.ccl22.trg


cat $sigahn_src $TRAIN_FILE".src"  > $TRAIN_SRC_FILE
cat $sigahn_trg  $TRAIN_FILE".trg" > $TRAIN_TRG_FILE


DEV_SRC_FILE=./track1_data/dev/yaclc-csc_dev.src
DEV_TRG_FILE=./track1_data/dev/yaclc-csc_dev.lbl


if [ ! -f $DATA_DIR"/train.pkl" ]; then
    python ./data_preprocess.py \
    --source_dir $TRAIN_SRC_FILE \
    --target_dir $TRAIN_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/train.pkl" \
    --data_mode "para" \
    --normalize "True"
fi

if [ ! -f $DATA_DIR"/dev.pkl" ]; then
    python ./data_preprocess.py \
    --source_dir $DEV_SRC_FILE \
    --target_dir $DEV_TRG_FILE \
    --bert_path $PRETRAIN_MODEL \
    --save_path $DATA_DIR"/dev.pkl" \
    --data_mode "lbl" \
    --normalize "True"
fi


BASE_MODEL=./roberta_pretrain/data_gen_1.pkl
base_tag=roberta_data_gen_1
# 第一种生成数据的方式预训练得到的模型
# BASE_MODEL=./roberta_pretrain/data_gen_2.pkl
## 第二种生成数据的方式预训练得到的模型
#base_tag=roberta_data_gen_2


# 训练
MODEL_DIR="./model/"$step"_"$tag"_"$base_tag
mkdir -p $MODEL_DIR

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train_mlm.py \
    --pretrained_model $PRETRAIN_MODEL \
    --train_path $DATA_DIR"/train.pkl" \
    --dev_path $DATA_DIR"/dev.pkl" \
    --lbl_path $DEV_TRG_FILE \
    --save_path $MODEL_DIR \
    --batch_size 32 \
    --load_path $BASE_MODEL \
    --tie_cls_weight True \
	--patience 5 \
    --tag $tag

# > $MODEL_DIR"/log.txt"
