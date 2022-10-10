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
char_smi = CharFuncs('./ccl2022/char_meta.txt')


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

    p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
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
        self.confusion_set_pinyin = pickle.load(open("./ccl2022/pinyin_confset.pkl", "rb"))
        self.confusion_set_shape = pickle.load(open("./ccl2022/other_confset.pkl", "rb"))
        self.same_pinyin_words = pickle.load(
            open("./ccl2022/chinese_homophone_word_simple.pkl", "rb"))

        self.pair_words = {}
        # 加入训练集中统计到的词对
        with open('./ccl2022/word_pairs.txt', "r") as f:
            text = f.read().strip().split("\n")
            for item_str in text:
                item_li = item_str.strip().split("\t")
                t, s = item_li[0].strip(), item_li[1].strip()
                if t in self.pair_words:
                    self.pair_words[t].add(s)
                else:
                    self.pair_words[t] = set()
                    self.pair_words[t].add(s)
        # 加入验证集中的混淆集
        with open('./ccl2022/add_confset.txt', "r") as f:
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

        vocab = pickle.load(open("./ccl2022/track1_vocab.pkl", "rb"))
        self.vocab = [s for s in vocab if s in vob and is_chinese(s)]
        print(len(self.vocab))
        self.read(self.infile, self.outfile)
        # self.write(self.corpus, self.outfile)

    def read(self, path, path_out):
        """
        :param path:
        :param path_out:
        :return:
        """
        print("reading now......")
        with open(path, encoding="UTF-8") as f, open(path_out, "w", encoding="UTF-8") as fw:
            i = 0
            while True:
                line = f.readline().strip()
                if not line:
                    break
                i += 1
                line = normalize_lower(line)
                new_line = self.replace_token(line)
                if new_line.strip() != line.strip():
                    fw.writelines(new_line.strip() + " " + line.strip() + "\n")
                if i % 10000 == 0:
                    print(i // 1000000)
        print("read finished.")

    def get_replace_token(self, char):
        """
        获取混淆字
        :param char:
        :return:
        """
        rep_char = char
        if np.random.rand(1) < 0.84:
            if char in self.confusion_set_pinyin and len(self.confusion_set_pinyin[char]) != 0:
                token_conf_set = self.confusion_set_pinyin[char]
                idx = np.random.randint(0, len(token_conf_set))
                rep_char = list(token_conf_set)[idx]
        elif np.random.rand(1) < 0.97:
            if is_chinese(char):
                idx = np.random.randint(0, len(self.vocab))
                rep_char = self.vocab[idx]
        else:
            if char in self.confusion_set_shape and len(self.confusion_set_shape[char]) != 0:
                token_conf_set = self.confusion_set_shape[char]
                idx = np.random.randint(0, len(token_conf_set))
                rep_char = list(token_conf_set)[idx]
        return rep_char

    def get_replace_word(self, word):
        """
        获取同音词
        :param word:
        :return:
        """
        rep_word = word
        if np.random.rand(1) < 0.43:
            w_pinyin_li = lazy_pinyin(word)
            w_pinyin_str = "_".join(w_pinyin_li)
            if w_pinyin_str in self.same_pinyin_words and len(self.same_pinyin_words[w_pinyin_str]) != 0:
                token_conf_set = self.same_pinyin_words[w_pinyin_str]
                idx = np.random.randint(0, len(token_conf_set))
                rep_word = list(token_conf_set)[idx]
        else:
            if word in self.pair_words and len(self.pair_words[word]) != 0:
                token_conf_set = self.pair_words[word]
                idx = np.random.randint(0, len(token_conf_set))
                rep_word = list(token_conf_set)[idx]
        return rep_word

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
        words = list(jieba.cut(line))
        # i 所在的词,以及这个词所包含的位置
        idx2word = {}
        start = 0
        for w in words:
            for i in range(start, start + len(w)):
                idx2word[i] = (w, start, start + len(w))
            start += len(w)

        for j in range(num // options.sep + 1):
            if options.sep * j == num:
                continue
            i = np.random.randint(options.sep * j, min((j + 1) * options.sep, num))
            if np.random.rand(1) < 0.43:
                rep_char = self.get_replace_token(tokens[i])
                if rep_char != tokens[i]:
                    tokens[i] = rep_char
                    count += 1
            else:
                ori_word, s, e = idx2word[i]
                rep_word = self.get_replace_word(ori_word)
                if rep_word != ori_word:
                    tokens[s:e] = list(rep_word)
                    count += 1

            if count >= up_num:
                break
        return "".join(tokens)


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
