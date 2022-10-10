

gpu=3
PRETRAIN_MODEL=/home/plm_models/chinese-roberta-wwm-ext
DATA_DIR=../track1/track1_data/test/



MODEL_PATH0=./2_bert_data_gen_2/sighan_hsk_split-epoch-1.pt
MODEL_PATH1=./2_macbert_data_gen_2/sighan_hsk_error_word_one_split-epoch-1.pt
MODEL_PATH2=./2_roberta_data_gen_2/sighan_hsk_error_word_one_split-epoch-4.pt
MODEL_PATH3=./2_bert_data_gen_1/sighan_hsk_split-epoch-1.pt
MODEL_PATH4=./3_macbert_data_gen_2/sighan-epoch-6.pt
MODEL_PATH5=./3_roberta_data_gen_2/sighan-epoch-7.pt
MODEL_PATH6=./3_bert_data_gen_2/sighan-epoch-2.pt


TEST_SRC_FILE=../track1/track1_data/test/yaclc-csc_test.src
TAG=track1_test


# 多次迭代，但是每次迭代仅仅选择，统计语言模型概率更高的
TMP_INPUT=../track1/track1_data/test/yaclc-csc_test
cp $TEST_SRC_FILE $TMP_INPUT


for ((i=1; i<=2; i++))
do

j=$(($i-1))

python ../track1/data_preprocess.py \
--source_dir $TMP_INPUT \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".pkl" \
--data_mode "lbl" \
--normalize "True"


CUDA_VISIBLE_DEVICES=$gpu python ../track1/decode_mlm_em.py \
    --pretrained_model $PRETRAIN_MODEL \
    --test_path $DATA_DIR"/test_"$TAG".pkl" \
    --model_path $MODEL_PATH0","$MODEL_PATH1","$MODEL_PATH2","$MODEL_PATH3","$MODEL_PATH4","$MODEL_PATH5","$MODEL_PATH6 \
    --save_path $DATA_DIR"/yaclc-csc-test.lbl" \
    --batch_size 8


cp $DATA_DIR"/yaclc-csc-test.lbl.pre" $DATA_DIR"/yaclc-csc_test_"$i".pre"

# 剔除一些非音近形近修改，成语匹配纠错
python postprocess_test.py $DATA_DIR"/yaclc-csc_test_"$i".pre" $DATA_DIR"/yaclc-csc-test_"$i".post"

# 通过语言模型，选择上一次修改或者最新的修改
if [ ! -e $DATA_DIR"/yaclc-csc-test_"$j".out" ]; then
  cp $DATA_DIR"/yaclc-csc-test_"$i".post" $DATA_DIR"/yaclc-csc-test_"$i".out"
else
  python select_best.py  $DATA_DIR"/yaclc-csc-test_"$j".out" $DATA_DIR"/yaclc-csc-test_"$i".post" $DATA_DIR"/yaclc-csc-test_"$i".out"
fi

# 转成输入需要的格式
python trans_format.py $DATA_DIR"/yaclc-csc-test_"$i".out" $TMP_INPUT

done


# 模型集成后的结果再经过验证集训练的模型进行纠错，相当于先改通用错误，再改领域错误
MODEL_PATH_DEV=./3_roberta_data_gen_2_dev/dev-epoch-8.pt


for ((i=3; i<4; i++))
do

j=$(($i-1))

python ../track1/data_preprocess.py \
--source_dir $TMP_INPUT \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".pkl" \
--data_mode "lbl" \
--normalize "True"


CUDA_VISIBLE_DEVICES=$gpu python ../track1/decode_mlm_em.py \
    --pretrained_model $PRETRAIN_MODEL \
    --test_path $DATA_DIR"/test_"$TAG".pkl" \
    --model_path $MODEL_PATH_DEV \
    --save_path $DATA_DIR"/yaclc-csc-test.lbl" \
    --batch_size 8


cp $DATA_DIR"/yaclc-csc-test.lbl.pre" $DATA_DIR"/yaclc-csc_test_"$i".pre"

python postprocess_test.py $DATA_DIR"/yaclc-csc_test_"$i".pre" $DATA_DIR"/yaclc-csc-test_"$i".post"

if [ ! -e $DATA_DIR"/yaclc-csc-test_"$j".out" ]; then
  cp $DATA_DIR"/yaclc-csc-test_"$i".post" $DATA_DIR"/yaclc-csc-test_"$i".out"
else
  python select_best.py  $DATA_DIR"/yaclc-csc-test_"$j".out" $DATA_DIR"/yaclc-csc-test_"$i".post" $DATA_DIR"/yaclc-csc-test_"$i".out"
fi

python trans_format.py $DATA_DIR"/yaclc-csc-test_"$i".out" $TMP_INPUT

done


rm -rf track1_test.zip
rm -rf yaclc-csc-test.lbl

python process_test.py $DATA_DIR"/yaclc-csc-test_3.out"

zip track1_test.zip yaclc-csc-test.lbl