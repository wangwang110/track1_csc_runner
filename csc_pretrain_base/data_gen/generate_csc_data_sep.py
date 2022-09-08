# -*- coding: UTF-8 -*-


import os
import pickle
import re
from optparse import OptionParser
import numpy as np
from strTools import uniform, is_chinese
from char_smi import CharFuncs
from pypinyin import pinyin, lazy_pinyin, Style
import jieba

# 用于衡量字形相似度
char_smi = CharFuncs('/home/MuCGEC/scorers/ChERRANT/data/char_meta.txt')


def pinyin_similar(src_pinyin, tgt_pinyin):
    for a in src_pinyin:
        for b in tgt_pinyin:
            if a + "g" == b or b + "g" == a:
                # 前后鼻音不分
                return True
            if a.replace("zh", "z").replace("sh", "s").replace("ch", "c") == b \
                    or b.replace("zh", "z").replace("sh", "s").replace("ch", "c") == a:
                # 平翘舌不分
                return True


def is_pinyin(src_char, tgt_char):
    p_threshold = 0.7
    # s_threshold = 0.42
    # try:
    #     v_sim = char_smi.shape_similarity(src_char, tgt_char)
    # except:
    #     print(src_char + "====" + tgt_char)  # 不在字形库
    #     return None
    try:
        p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
    except:
        print("不是汉字")
        return False
    src_pinyin = set(pinyin(src_char, style=Style.NORMAL, heteronym=True)[0])
    tgt_pinyin = set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])

    if len(src_pinyin & tgt_pinyin) != 0 or p_sim >= p_threshold:
        return True
    if pinyin_similar(src_pinyin, tgt_pinyin):
        return True
    #
    # if v_sim < s_threshold:
    #     print(src_char, tgt_char)
    #     print(v_sim)
    return False


vob = set()
with open("/home/plm_models/chinese_L-12_H-768_A-12/vocab.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        vob.add(line.strip())


def normalize_lower(text):
    """
    # 非汉字全部半角化？
    # 统一起来会比较好
    :param text:
    :return:
    """
    text = text.strip().lower()
    text = re.sub("\s+", "", text)
    text = re.sub("\ue40c", "", text)
    return text


class GenerateCSC:
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self.corpus = []
        self.confusion_set_pinyin = pickle.load(open("./track1_data/ccl2022/pinyin_confset.pkl", "rb"))
        self.confusion_set_shape = pickle.load(open("./track1_data/ccl2022/other_confset.pkl", "rb"))

        # 加入验证集中的混淆集
        with open('./track1_data/ccl2022/add_confset.txt', "r") as f:
            text = f.read().strip().split("\n")
            for item_str in text:
                item_li = item_str.strip().split("\t")
                s, t = item_li[0].strip(), item_li[1].strip()
                if is_pinyin(s, t):
                    if t in self.confusion_set_pinyin:
                        self.confusion_set_pinyin[t].add(s)
                    else:
                        self.confusion_set_pinyin[t] = set()
                        self.confusion_set_pinyin[t].add(s)
                else:
                    if t in self.confusion_set_shape:
                        self.confusion_set_shape[t].add(s)
                    else:
                        self.confusion_set_shape[t] = set()
                        self.confusion_set_shape[t].add(s)

        vocab = pickle.load(open("./track1_data/ccl2022/track1_vocab.pkl", "rb"))
        self.vocab = [s for s in vocab if s in vob and is_chinese(s)]
        print(len(self.vocab))
        self.read(self.infile)
        self.write(self.corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        with open(path, encoding="UTF-8") as f:
            i = 0
            for line in f.readlines():
                i += 1
                line = normalize_lower(line)
                new_line = self.replace_token(line)
                self.corpus.append([line, new_line])
        print("read finished.")

    def get_replace_token(self, char):
        """
        获取混淆字
        :param char:
        :return:
        """
        rep_char = char
        if np.random.rand(1) < 0.8:
            if char in self.confusion_set_pinyin and len(self.confusion_set_pinyin[char]) != 0:
                token_conf_set = self.confusion_set_pinyin[char]
                idx = np.random.randint(0, len(token_conf_set))
                rep_char = list(token_conf_set)[idx]
        elif np.random.rand(1) < 0.95:
            idx = np.random.randint(0, len(self.vocab))
            rep_char = self.vocab[idx]
        else:
            if char in self.confusion_set_shape and len(self.confusion_set_shape[char]) != 0:
                token_conf_set = self.confusion_set_shape[char]
                idx = np.random.randint(0, len(token_conf_set))
                rep_char = list(token_conf_set)[idx]
        return rep_char

    def replace_token(self, line):
        """
        每隔10个汉字插入一个错误
        :param line:
        :return:
        """
        num = len(line)
        tokens = list(line)
        up_num = int(num * options.ratio)  # 最多插入的错误个数占比
        count = 0
        for j in range(num // options.sep + 1):
            if options.sep * j == num:
                continue
            i = np.random.randint(options.sep * j, min((j + 1) * options.sep, num))
            rep_char = self.get_replace_token(tokens[i])
            if rep_char != tokens[i]:
                tokens[i] = rep_char
                count += 1
            if count >= up_num:
                break
        return "".join(tokens)

    def write(self, list, path):
        print("writing now......")
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for item in list:
            line1, line2 = item
            file.writelines(line2.strip() + " " + line1.strip() + "\n")
        file.close()
        print("writing finished")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--path", dest="path", default="", help="path file")
    parser.add_option("--input", dest="input", default="", help="input file")
    parser.add_option("--output", dest="output", default="", help="output file")
    parser.add_option("--ratio", type=float, default=0.25, help="error ratio")
    parser.add_option("--confuse_ratio", type=float, default=0.9, help="error ratio")
    parser.add_option("--seed", type=int, default=10, help="random seed")
    parser.add_option("--confpath", dest="confpath", default="", help="confpath file")
    parser.add_option("--sep", type=int, default=10, help="sep num")

    (options, args) = parser.parse_args()
    path = options.path
    input = options.input
    output = options.output

    np.random.seed(options.seed)
    GenerateCSC(infile=input, outfile=output)
    print("All Finished.")