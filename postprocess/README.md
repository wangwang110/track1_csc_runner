
## 拼写纠错模型的集成以及后处理

集成：保持相同的预测词表，以概率集成方式集成7个模型

后处理：

（1）为了能够修改更多的错误，尤其是连续错误，进行多次前向，迭代修改，通过统计语言模型选择是否保留修改

（2）为了减少误修改，对非音近，形近的修改，进行还原。

（3）为了能更好的处理成语错误，增加了成语匹配纠错模块，若能够收集到足够多的实体，该方法同样使用实体纠错。

（4）为了能够更好的利用验证集，模型集成后的结果再经过验证集训练的模型进行纠错，相当于先改通用错误，再改领域错误。


模型集成以及后处理，最后的运行结果是句子级F1为81.12：

`
sh ./decode_mlm_em_multi.sh
`

### 相关模型下载

成语匹配纠错相关文件：
链接: https://pan.baidu.com/s/1xDUZ8vQ0Bloj-zmICkPQZg

提取码: vr4u


待集成模型：
链接: https://pan.baidu.com/s/1JeKrv0Qv2t1SasEHI1rJEw

提取码: kdwx 