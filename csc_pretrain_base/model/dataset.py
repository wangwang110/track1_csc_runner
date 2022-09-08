from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LargeTrainDataset(Dataset):
    def __init__(self, filepath):
        number = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for _ in tqdm(f, desc="load training dataset"):
                number += 1
        self.number = number
        # 获取长度
        self.fopen = open(filepath, 'r')

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        line = self.fopen.__next__()
        line = line.strip()
        if len(line.split(" ")) == 2:
            pairs = line.split(" ")
            elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
        elif len(line.split("|||")) == 2:
            pairs = line.split("|||")
            elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
        elif len(line.split("\t")) == 2:
            pairs = line.split("\t")
            elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
        else:
            pairs = line.split(" ")
            elem = {'input': "", 'output': pairs[0]}
        return elem


class BertDataset(Dataset):
    def __init__(self, dataset):
        # self.tokenizer = tokenizer
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        data = self.dataset[index]
        return data

#
# def construct(filename):
#     list = []
#     with open(filename, encoding='utf-8') as f:
#         for line in f:
#             try:
#                 line = line.strip()
#                 if len(line.split(" ")) == 2:
#                     pairs = line.split(" ")
#                     elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
#                 elif len(line.split("|||")) == 2:
#                     pairs = line.split("|||")
#                     elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
#                 elif len(line.split("\t")) == 2:
#                     pairs = line.split("\t")
#                     elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
#                 else:
#                     pairs = line.split(" ")
#                     elem = {'input': "", 'output': pairs[0]}
#                 list.append(elem)
#             except Exception as e:
#                 print(e)
#                 continue
#     return list


def construct(filename):
    list = []
    with open(filename, encoding='ISO-8859-1') as f:
        for line in f:
            try:
                # 将读取出来的数据先用ISO-8859-1格式给它编码，然后通过utf-8给它解码
                line = line.encode('ISO-8859-1').decode('utf-8')
            except UnicodeError as e:
                print(e)
                continue

            try:
                line = line.strip()
                if len(line.split(" ")) == 2:
                    pairs = line.split(" ")
                    elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
                elif len(line.split("|||")) == 2:
                    pairs = line.split("|||")
                    elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
                elif len(line.split("\t")) == 2:
                    pairs = line.split("\t")
                    elem = {'input': pairs[0].strip(), 'output': pairs[1].strip()}
                else:
                    pairs = line.split(" ")
                    elem = {'input': "", 'output': pairs[0]}
                list.append(elem)
            except Exception as e:
                print(e)
                continue
    return list


# with open('test.txt', 'r', encoding='ISO-8859-1') as f:
#     for i in tqdm(f):
#         # 打印出来数据是ISO-8859-1编码
#         # print(i)
#         # 此处可能还是会因为数据中的特殊字符导致报错
#         try:
#             # 将读取出来的数据先用ISO-8859-1格式给它编码，然后通过utf-8给它解码
#             x = i.encode('ISO-8859-1').decode('utf-8')
#         except UnicodeError as e:
#             print(e)
#             # 跳过出错的数据
#             x = ''
#
#         if x == '':
#             print(x)
#             with open('test_new.txt', 'a', encoding='utf-8') as f:
#                 f.write(x)


def construct_pretrain(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        try:
            line = line.replace("\n", "")
            pairs = line.split(" ")
            elem = {'input': "", 'output': pairs[0]}
            list.append(elem)
        except Exception as e:
            print(e)
            continue
    return list


def construct_ner(filename):
    f = open(filename, encoding='utf8')
    ner_li = open("/data_local/TwoWaysToImproveCSC/BERT/data/merge_train_ner_tag.txt", encoding='utf8').readlines()

    list = []
    i = 0
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        ner_ids_str = ner_li[i]
        # print(ner_ids_str)
        try:
            elem = {'input': pairs[0], 'output': pairs[1], 'output_ner': ner_ids_str}
            list.append(elem)
        except Exception as e:
            print(e)
            continue
        i += 1
    return list


def singleconstruct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        if (len(pairs[0]) != len(pairs[1])):
            continue
        elem = {'input': pairs[1], 'output': pairs[1]}
        list.append(elem)
    return list


def testconstruct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        elem = {'input': pairs[0], 'output': ""}
        list.append(elem)
    return list


def cc_testconstruct(filename):
    f = open(filename, encoding='utf8')
    list = []
    for line in f:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        elem = {'output': pairs[0], 'input': pairs[1]}
        list.append(elem)
    return list


def li_testconstruct(sent_li):
    list = []
    for line in sent_li:
        line = line.replace("\n", "")
        pairs = line.split(" ")
        elem = {'input': pairs[0], 'output': ""}
        list.append(elem)
    return list
