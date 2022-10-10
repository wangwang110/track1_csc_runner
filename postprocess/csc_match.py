# -*- coding: utf-8 -*-


import kenlm
import re
import os
import jieba
import pickle
from pypinyin import pinyin, lazy_pinyin, Style
from utils import confusion_set_pinyin, confusion_set_shape, is_pinyin, is_shape


class CSCmatch:
    def __init__(self):
        # 成语匹配字典
        self.cy_dict = pickle.load(open("./chengyu_match/match_cy.pkl", "rb"))
        self.cy_set = pickle.load(open("./chengyu_match/cy_set.pkl", "rb"))

        # self.model = kenlm.Model("/data_local/slm/ccl2022_track4_train.bin")
        self.model = kenlm.Model("./chengyu_match/chinese_csc_char.bin")
        print("model load success !!")

    def tokenize(self, texts):
        """
        分词 jieba.cut("我来到北京清华大学", cut_all=False)
        :param texts:
        :return:
        """
        res = []
        for text in texts:
            cut_gen = jieba.cut(text, cut_all=False)
            res.append(list(cut_gen))
        return res

    def getscores(self, sents):
        """
        获得语言模型的得分
        :param sents:
        :return:
        """
        res = []
        for text in sents:
            score = self.model.score(text.strip(), bos=False, eos=False)
            res.append(score)
        return res

    def cy_candidates(self, query_tran='高瞻远瞩'):
        """
        匹配成语
        :param query_tran:
        :return:
        """
        candidates = set()
        num = len(query_tran)
        for i in range(num):
            tokens = list(query_tran)
            tokens[i] = "_"
            key = "".join(tokens)
            if key in self.cy_dict:
                candidates = candidates | self.cy_dict[key]

        #
        final_res = []
        for match_str in candidates:
            for s, t in zip(query_tran, match_str):
                if s != t:
                    if self.is_mix(s, t):
                        final_res.append(match_str)
                    break
        return final_res

    def cy_correct(self, texts, change_li_all):
        """
        :param text:
        :param no_change_li:
        :return:
        """
        match_cy_li = []
        pos_cy_li = []

        for text, change_li in zip(texts, change_li_all):
            query_text = text
            words = self.tokenize([text])[0]
            # 相邻两个组成4字候选成语
            # jieba分词？
            s = 0
            correct = {}
            cy_li = []
            num = len(words)
            for i in range(num):
                for d in range(1, 5, 1):
                    j = i + d
                    if j <= num:
                        word_str = "".join(words[i:j])

                        if len(word_str) == 4 and word_str in self.cy_set:
                            correct[word_str] = word_str
                            # 正确的匹配到的
                            continue
                        if len(word_str) == 4 and not self.is_no_change(s, word_str, change_li):
                            w_src = word_str
                            match_trgs = self.cy_candidates(w_src)
                            if len(match_trgs) != 0:
                                w_trg = self.cy_lm_correct_part(match_trgs, i, j, words)
                                if w_src != w_trg:
                                    correct[w_src] = w_trg
                                    cy_li.extend([s + p for p in range(len(word_str))])
                s += len(words[i])

            pos_cy_li.append(cy_li)
            match_cy_li.append(correct)
        return match_cy_li, pos_cy_li

    def cy_correct_sent(self, text):
        """
        :param text:
        :param no_change_li:
        :return:
        """

        query_text = text
        words = self.tokenize([text])[0]
        # 相邻两个组成4字候选成语
        # jieba分词？
        s = 0
        correct = {}
        num = len(words)
        for i in range(num):
            for d in range(1, 5, 1):
                j = i + d
                if j <= num:
                    word_str = "".join(words[i:j])
                    if len(word_str) == 4 and word_str in self.cy_set:
                        correct[word_str] = word_str
                        # 正确的匹配到的
                        continue
                    if len(word_str) == 4:
                        w_src = word_str
                        match_trgs = self.cy_candidates(w_src)
                        if len(match_trgs) != 0:
                            w_trg = self.cy_lm_correct_part(match_trgs, i, j, words)
                            if w_src != w_trg:
                                correct[w_src] = w_trg
                                query_text = "".join(words[:i]) + w_trg + "".join(words[j:])

            s += len(words[i])
        return query_text

    def is_mix(self, s, t):
        """
        是否考虑多音字
        :param src:
        :param trgs:
        :return:
        """

        if (t in confusion_set_pinyin and s not in confusion_set_pinyin[t]) \
                and (t in confusion_set_shape and s not in confusion_set_shape[t]) \
                and not is_pinyin(s, t) and not is_shape(s, t):
            return False
        return True

    def cy_lm_correct_part(self, match_trgs, start, end, words, name="cy"):
        """
        查找相似成语
        :param src_text:
        :return:
        """
        # 前后个五个字
        # A B C D EF AB  C D E F G
        # 0 1 2 3  4  5  6 7 8 9 10

        # A B C D E F A B C D E  F   G
        # 0 1 2 3 4 5 6 7 8 9 10 11 12

        w_src_str = "".join(words[start:end])
        w_src = " ".join(list(w_src_str))

        src_sent = "".join(words)

        low = max(0, len("".join(words[:start])) - 4)  # 4  # start - 4
        high = min(len(src_sent), len("".join(words[:end])) + 4)  # 8  #  end + 4

        src_text = " ".join(list(src_sent)[low:high])
        #
        # print(w_src)
        # print(src_text)

        candidates = [src_text]
        candidate_words = [w_src_str]
        for match_trg in match_trgs:
            match_str = " ".join(list(match_trg))
            tmp_text = src_text.replace(w_src, match_str)
            candidates.append(tmp_text)
            candidate_words.append(match_trg)
        candidate_scores = self.getscores(candidates)
        item = sorted(zip(candidate_words, candidate_scores), key=lambda s: s[1], reverse=True)
        # print(w_src)
        # print(item)
        if name == 'cy':
            return item[0][0]
        elif name == 'ci':
            if item[0][1] - item[1][1] > -0.3 * item[0][1]:
                return item[0][0]
            else:
                return w_src


if __name__ == '__main__':
    obj = CSCmatch()
    paths = [
        "../track1/track1_data/test/yaclc-csc-test.lbl.pre",
    ]
    for path in paths:
        all_texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sent = line.strip()
                all_texts.append(sent)

        path_out = "../track1/track1_data/test/yaclc-csc-test.lbl.out"
        with open(path_out, "w", encoding="utf-8") as fw:
            num = len(all_texts)
            for i in range(num):
                text = all_texts[i]
                pre = obj.cy_correct_sent(text)
                if text != pre:
                    print(text)
                    print(pre)
                    fw.write(pre + '\n')
                else:
                    fw.write(text + "\n")
