# track1_csc_runner
ccl2022赛道一中文拼写检查

1. csc_pretrain_base中是预训练以及数据生成的代码,具体如何使用，可参考文件夹中相关文件

2. track1中是微调代码，在比赛给出的baseline的基础上做了一些修改，具体如何使用，可参考文件夹中相关文件

3. postprocess是后处理代码，具体如何使用，可参考文件夹中相关文件


想快速得出比赛结果，直接使用postprocess文件夹中的代码

`
sh ./decode_mlm_em_multi.sh
`