# csc_pretrain_base
将生成的数据用于预训练拼写检查模型，以roberta预训练模型为例，训练脚本


`
sh ./pretrain_roberta_process.sh
`

其中，预训练模型下载：

bert：https://huggingface.co/bert-base-chinese

roberta：https://huggingface.co/hfl/chinese-roberta-wwm-ext

macbert： https://huggingface.co/hfl/chinese-macbert-base

生成数据代码在data_gen,具体如何使用，可参考文件夹中相关文件

