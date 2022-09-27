
# 第三阶段的精调
PRETRAIN_MODEL=/data_local/plm_models/chinese-roberta-wwm-ext/

tag="dev"
step=3
CUDA_DEVICE=5

################# 不修改 #################

DATA_DIR="./data/"$tag
mkdir -p $DATA_DIR


TRAIN_SRC_FILE=./track1_data/dev/ccl_trck1_dev.src
TRAIN_TRG_FILE=./track1_data/dev/ccl_trck1_dev.trg

DEV_SRC_FILE=./track1_data/sighan_data/dev_ae.src
DEV_TRG_FILE=./track1_data/sighan_data/dev_ae.lbl


DEV_SRC_FILE=./track1_data/sighan_data/dev_ae.src
DEV_TRG_FILE=./track1_data/sighan_data/dev_ae.lbl


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



BASE_MODEL=./roberta_finetune/2_all_roberta_data_gen_2/merge_one_wke_uniq-epoch-1.pt
model_name="roberta_data_gen_2"

# 训练
MODEL_DIR="./roberta_finetune/"$step"_"$tag"_"$model_name
mkdir -p $MODEL_DIR/

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train_mlm.py \
    --pretrained_model $PRETRAIN_MODEL \
    --train_path $DATA_DIR"/train.pkl" \
    --dev_path $DATA_DIR"/dev.pkl" \
    --lbl_path $DEV_TRG_FILE \
    --save_path $MODEL_DIR \
    --batch_size 32 \
    --load_path $BASE_MODEL \
    --tie_cls_weight True \
    --tag $tag
    # > $MODEL_DIR"/log.txt"
