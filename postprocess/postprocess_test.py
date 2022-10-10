# coding: utf-8

"""
@File    : analysis_test.py.py
@Time    : 2022/7/18 16:16
@Author  : liuwangwang
@Software: PyCharm
"""

import sys
from utils import confusion_set_pinyin, confusion_set_shape, is_pinyin, is_shape
from csc_match import CSCmatch

base = "/home/track1_csc_runner/csc_pretrain_base/data_gen/ccl2022/"
obj = CSCmatch()

# 加入验证集中的混淆集
with open(base + '/add_confset.txt', "r") as f:
    text = f.read().strip().split("\n")
    for item_str in text:
        item_li = item_str.strip().split("\t")
        s, t = item_li[0].strip(), item_li[1].strip()
        if is_pinyin(s, t):
            if t in confusion_set_pinyin:
                confusion_set_pinyin[t].add(s)
            else:
                confusion_set_pinyin[t] = set()
                confusion_set_pinyin[t].add(s)
        else:
            if t in confusion_set_shape:
                confusion_set_shape[t].add(s)
            else:
                confusion_set_shape[t] = set()
                confusion_set_shape[t].add(s)

paths = [
    "../track1/track1_data/test/yaclc-csc_test.txt",
    "../track1/track1_data/test/yaclc-csc_test.txt"
]

pre_path = sys.argv[1]
path_out = sys.argv[2]

all_srcs = [line.strip() for line in open(paths[0]).readlines()]
all_pres = [line.strip() for line in open(pre_path).readlines()]
trgs = [line.strip() for line in open(paths[1]).readlines()]

lens = []
res = []
for src, pre0, trg in zip(all_srcs, all_pres, trgs):

    pre = obj.cy_correct_sent(pre0)

    if pre != pre0:
        print("成语匹配纠错")
        print(pre0)
        print(pre)

    tokens = []
    words = list(src)
    i = 0
    for s, t in zip(list(src), list(pre)):
        if s != t:
            if s in ["您", "你"]:
                tokens.append(s)
            elif i > 0 and src[i] + src[i - 1] == pre[i - 1] + pre[i]:
                tokens.append(t)
            elif i + 1 < len(pre) and src[i] + src[i + 1] == pre[i + 1] + pre[i]:
                tokens.append(t)
            elif (t in confusion_set_pinyin and s not in confusion_set_pinyin[t]) \
                    and (t in confusion_set_shape and s not in confusion_set_shape[t]) \
                    and not is_pinyin(s, t) and not is_shape(s, t):
                print(src)
                print(pre0)
                print(s, t)
                print("=============")
                tokens.append(s)
            else:
                tokens.append(t)
        else:
            tokens.append(s)
        i += 1

    # 怎么使用统计语言模型
    pre = "".join(tokens)
    num = len(pre)
    for i in range(num):
        s = src[i]
        t = pre[i]
        if s != t:
            # 替换回原来的是否更好
            cor_tokens = list(pre)
            pre_tokens = list(pre)
            cor_tokens[i] = s
            scores = obj.getscores([" ".join(cor_tokens), " ".join(pre_tokens)])
            if scores[1] > scores[0] - 4.5:
                tokens[i] = t
            else:
                # 替换回原来
                print("保留原有的句子")
                print(src)
                print(pre0)
                tokens[i] = s
                print(scores)
                print(s, t)
                print("############")
    res.append("".join(tokens))

with open(path_out, "w") as fw:
    for ss in res:
        fw.write(ss + "\n")
