
## 文件说明


### 生成数据的方法

第一种： 按照一定的比例，随机替换句子中正确的汉字为对应的易混淆汉字，以0.25为例

```
python generate_csc_data_process.py --input ./ccl2022/test_gen.trg --output ./ccl2022/test_gen.out --ratio 0.25
```

第二种：每隔多少个汉字，随机替换句子中正确的汉字为对应的易混淆汉字，以每隔10个为例，这种做法无法构造出连续错误，因此引入同音词与错误词对，即可能直接替换某个词


```
python generate_csc_data_word.py --input ./ccl2022/test_gen.trg --output ./ccl2022/test_gen.out --sep 10
```

代码中需要用到的文件，可从 [这里](https://pan.baidu.com/s/1fWBGkSzcsswYCVWGgorHNg?pwd=q8m3) 下载，放在ccl2022目录下


### 生成的数据

最后生成的伪数据可从 [这里](https://pan.baidu.com/s/1YHSzWKUsmR5fhlKPxhHB5w?pwd=eb3c) 下载

(1) 按照第一种策略生成一份数据 1300w_process.csc

(2) 将数据重采样三分，按照第二种策略生成一份数据 1300w_word.csc
