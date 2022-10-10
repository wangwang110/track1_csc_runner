PRETRAIN_MODEL=/home/plm_models/chinese-roberta-wwm-ext

## 集成模型结果测试

DATA_DIR=./track1_data/test/
TEST_SRC_FILE=./track1_data/test/yaclc-csc_test.src
TAG=track1_test

python ./data_preprocess.py \
--source_dir $TEST_SRC_FILE \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".pkl" \
--data_mode "lbl" \
--normalize "True"

rm -rf $DATA_DIR"/track1_test.zip"
rm -rf $DATA_DIR"/yaclc-csc-test.lbl"

gpu=1


# 加上dev阶段是有效的
MODEL_PATH0=./roberta_finetune/3_sighan_roberta_data_gen_1/sighan-epoch-23.pt
MODEL_PATH1=./roberta_finetune/3_sighan_roberta_data_gen_2/sighan-epoch-23.pt
MODEL_PATH2=./roberta_finetune/2_hsk_roberta_data_gen_1/hsk_error_word_one_uniq-epoch-1.pt
MODEL_PATH3=./roberta_finetune/2_hsk_roberta_data_gen_2/hsk_error_word_one_uniq-epoch-4.pt
# 76.63
# 77.53 + 2*dev


CUDA_VISIBLE_DEVICES=$gpu python decode_mlm_em.py \
    --pretrained_model $PRETRAIN_MODEL \
    --test_path $DATA_DIR"/test_"$TAG".pkl" \
    --model_path $MODEL_PATH0","$MODEL_PATH1","$MODEL_PATH2","$MODEL_PATH3 \
    --save_path $DATA_DIR"/yaclc-csc-test.lbl" \
    --batch_size 8

# python utils.py $DATA_DIR"/yaclc-csc-test.lbl" $DATA_DIR"/yaclc-csc_test_new.lbl"

cd $DATA_DIR
zip track1_test.zip yaclc-csc-test.lbl

