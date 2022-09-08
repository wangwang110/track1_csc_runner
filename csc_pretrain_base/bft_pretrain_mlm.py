# -*- coding: UTF-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import string
from torch.optim import Adam
from data.getF1 import sent_metric_correct
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from model import BertFineTuneMac, construct, BertDataset

"""
分词：不用bert分词，直接一个字符是一个词
加入多任务学习：每个token是否有错（二分类），结果并无提升
loss修改：换用cpo_loss, combineloss(更加关注错误位置)，结果并无提升
"""

# def is_chinese(usen):
#     """判断一个unicode是否是汉字"""
#     for uchar in usen:
#         if '\u4e00' <= uchar <= '\u9fa5':
#             continue
#         else:
#             return False
#     else:
#         return True


print(torch.cuda.is_available())


class Trainer:
    def __init__(self, bert, optimizer, tokenizer, device):
        self.model = bert
        self.optim = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = 128
        # self.confusion_set = readAllConfusionSet('./save/confusion.file')

        # bert的词典
        self.vob = {}
        with open(bert_path + "vocab.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vob.setdefault(i, line.strip())
        #
        # # 动态掩码,sighan13略好，cc略差
        # vocab = pickle.load(open("./save/wiki_vocab.pkl", "rb"))
        # self.vocab = [s for s in vocab if s in self.vob.values() and is_chinese(s)]

    def train(self, train, epoch):
        self.model.train()
        total_loss = 0
        i = 0
        for batch in train:
            i += 1
            # if "pretrain" in args.task_name: # 动态掩码
            #     generate_srcs = self.replace_token(batch['output'])
            #     batch['input'] = generate_srcs
            inputs, outputs = self.help_vectorize(batch)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :self.max_len], \
                                                    inputs['token_type_ids'][:, :self.max_len], \
                                                    inputs['attention_mask'][:, :self.max_len]
            output_ids, output_token_label = outputs['input_ids'][:, :self.max_len], \
                                             outputs['token_labels'][:, :self.max_len]
            outputs = self.model(input_ids, input_tyi, input_attn_mask,
                                 text_labels=output_ids, det_labels=None, ignore=args.ignore_sep)
            c_loss = outputs[1].mean()
            total_loss += c_loss.item()
            self.optim.zero_grad()
            c_loss.backward()
            if i % 1000 == 0:
                print(c_loss.item())
            self.optim.step()
            if i % 30000 == 0:
                print("track_dev:")
                trainer.testSet_true(valid)
                if args.do_test:
                    print("13test:")
                    trainer.testSet_true(test)
                step_model_save_path = args.save_dir + '/epoch{0}_step{1}.pkl'.format(epoch, i)
                trainer.save(step_model_save_path)
                print("save model done! " + '/epoch{0}_step{1}.pkl'.format(epoch, i))
        return total_loss

    def save(self, name):
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def testSet_true(self, test):
        self.model.eval()
        all_srcs = []
        all_trgs = []
        all_pres = []
        for batch in test:
            all_srcs.extend(batch["input"])
            all_trgs.extend(batch["output"])
            inputs, outputs = self.help_vectorize(batch)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'][:, :self.max_len], \
                                                    inputs['token_type_ids'][:, :self.max_len], \
                                                    inputs['attention_mask'][:, :self.max_len]
            torken_prob, out = self.model(input_ids, input_tyi, input_attn_mask)
            out = out.argmax(dim=-1)
            num = len(batch["input"])
            for i in range(num):
                src = batch["input"][i]
                tokens = list(src)
                for j in range(min(len(tokens) + 1, self.max_len - 1)):
                    if out[i][j + 1] != input_ids[i][j + 1] and out[i][j + 1] not in [0, 100, 101, 102, 103]:
                        # and out[i][j + 1] not in [0, 100, 101, 102]:
                        # and out[i][j + 1] != 100
                        # structBert_large 经常预测为逗号，8024
                        ori_str = self.vob[input_ids[i][j + 1].item()]
                        rep_str = self.vob[out[i][j + 1].item()]
                        if j < len(tokens) and ori_str not in string.ascii_letters and len(ori_str) == len(rep_str):
                            tokens[j] = rep_str
                out_sent = "".join(tokens)
                # if out_sent != src:
                #     print(src)
                #     print(out_sent)
                #     print("=======================")
                all_pres.append(out_sent)

        acc, p, r, c_F1 = sent_metric_correct(all_srcs, all_pres, all_trgs)
        return acc, p, r, c_F1

    def text2vec(self, src, max_seq_length):
        """
        :param src:
        :return:
        """
        tokens_a = [a for a in src]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for j, tok in enumerate(tokens):
            if tok not in self.tokenizer.vocab:
                tokens[j] = "[UNK]"

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return input_ids, input_mask, segment_ids

    def help_vectorize(self, batch):
        """
        :param batch:
        :return:
        """
        src_li, trg_li = batch['input'], batch['output']
        max_seq_length = max([len(src) for src in src_li]) + 2
        inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        outputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

        for src, trg in zip(src_li, trg_li):
            input_ids, input_mask, segment_ids = self.text2vec(src, max_seq_length)
            inputs['input_ids'].append(input_ids)
            inputs['token_type_ids'].append(segment_ids)
            inputs['attention_mask'].append(input_mask)

            output_ids, output_mask, output_segment_ids = self.text2vec(trg, max_seq_length)
            outputs['input_ids'].append(output_ids)
            outputs['token_type_ids'].append(output_segment_ids)
            outputs['attention_mask'].append(output_mask)

        inputs['input_ids'] = torch.tensor(np.array(inputs['input_ids'])).to(self.device)
        inputs['token_type_ids'] = torch.tensor(np.array(inputs['token_type_ids'])).to(self.device)
        inputs['attention_mask'] = torch.tensor(np.array(inputs['attention_mask'])).to(self.device)

        outputs['input_ids'] = torch.tensor(np.array(outputs['input_ids'])).to(self.device)
        outputs['token_type_ids'] = torch.tensor(np.array(outputs['token_type_ids'])).to(self.device)
        outputs['attention_mask'] = torch.tensor(np.array(outputs['attention_mask'])).to(self.device)

        # 每个位置对应的分类，其中padding部分需要去掉
        token_labels = [
            [0 if inputs['input_ids'][i][j] == outputs['input_ids'][i][j] else 1
             for j in range(inputs['input_ids'].size()[1])] for i in range(len(batch['input']))]
        outputs["token_labels"] = torch.tensor(token_labels, dtype=torch.float32).to(self.device).unsqueeze(-1)
        return inputs, outputs


def setup_seed(seed):
    # set seed for CPU
    torch.manual_seed(seed)
    # set seed for current GPU
    torch.cuda.manual_seed(seed)
    # set seed for all GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Cancel acceleration
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)


def str2bool(strIn):
    if strIn.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif strIn.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(strIn)
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    import time

    # add arguments
    parser = argparse.ArgumentParser(description="choose which model")
    parser.add_argument('--task_name', type=str, default='bert_pretrain')
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--load_model', type=str2bool, nargs='?', const=False)
    parser.add_argument('--load_path', type=str, default='./save/13_train_seed0_1.pkl')
    parser.add_argument('--ignore_sep', type=str2bool, nargs='?', const=False)

    parser.add_argument('--do_train', type=str2bool, nargs='?', const=False)
    parser.add_argument('--train_data', type=str, default='../data/13train.txt')
    parser.add_argument('--do_valid', type=str2bool, nargs='?', const=False)
    parser.add_argument('--valid_data', type=str, default='../data/13valid.txt')
    parser.add_argument('--do_test', type=str2bool, nargs='?', const=False)
    parser.add_argument('--test_data', type=str, default='../data/13test.txt')

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--do_save', type=str2bool, nargs='?', const=False)
    parser.add_argument('--save_dir', type=str, default='../save')
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--noisy_tune', type=int, default=0)
    parser.add_argument('--bert_path', type=str, default="/home/plm_models/chinese_L-12_H-768_A-12/")

    args = parser.parse_args()
    task_name = args.task_name
    print("----python script: " + os.path.basename(__file__) + "----")
    print("----Task: " + task_name + " begin !----")
    print("----Model base: " + args.load_path + "----")
    print("----Train data: " + args.train_data + "----")
    print("----Batch size: " + str(args.batch_size) + "----")

    setup_seed(int(args.seed))
    start = time.time()

    device_ids = [i for i in range(int(args.gpu_num))]
    print(device_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_path = args.bert_path
    bert = BertForMaskedLM.from_pretrained(bert_path, return_dict=True)

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    config = BertConfig.from_pretrained(bert_path)
    model = BertFineTuneMac(bert, tokenizer, device, device_ids, is_correct_sent=True).to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_path))

    model = nn.DataParallel(model)
    if args.do_train:
        train = construct(args.train_data)
        train = BertDataset(train)
        train = DataLoader(train, batch_size=int(args.batch_size), shuffle=True)
        # all_update_setp = 2 * len(train)
        # print("update num:{}".format(all_update_setp))

    if args.do_valid:
        valid = construct(args.valid_data)
        valid = BertDataset(valid)
        valid = DataLoader(valid, batch_size=int(args.batch_size), shuffle=True)

    if args.do_test:
        test = construct(args.test_data)
        test = BertDataset(test)
        test = DataLoader(test, batch_size=int(args.batch_size), shuffle=False)

    optimizer = Adam(model.parameters(), float(args.learning_rate))

    trainer = Trainer(model, optimizer, tokenizer, device)
    max_f1 = 0
    best_epoch = 0
    if args.do_train:
        print("================testing================")
        if args.do_valid:
            print("track_dev:")
            valid_acc, valid_pre, valid_rec, valid_f1 = trainer.testSet_true(valid)
            if args.do_test:
                print("13test:")
                trainer.testSet_true(test)
        print("================testing================")

        for e in range(int(args.epoch)):
            print("epoch:{}".format(e + 1))

            train_loss = trainer.train(train, e + 1)

            if args.do_save:
                model_save_path = args.save_dir + '/epoch{0}.pkl'.format(e + 1)
                trainer.save(model_save_path)
                print("save model done!")

            if args.do_valid:
                print("track_dev:")
                valid_acc, valid_pre, valid_rec, valid_f1 = trainer.testSet_true(valid)
                if args.do_test:
                    print("track_test:")
                    trainer.testSet_true(test)
                max_f1 = valid_f1
            else:
                print(task_name, ",epoch {0},train_loss:{1}".format(e + 1, train_loss))

    if args.do_test:
        trainer.testSet_true(test)
