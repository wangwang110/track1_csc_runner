# coding: utf-8

"""
@Time    : 2022/3/16 17:53
@Author  : liuwangwang
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# class MarginLoss(nn.Module):
#     def __init__(self, margin=1.0, p=1, size_average=True):
#         super(MarginLoss, self).__init__()
#         self.margin = margin
#         self.size_average = size_average
#         self.p = p
#
#     def forward(self, prob, input_id, output_id):
#         """
#         :param input_id: B
#         :param output_id: B
#         :param model_logprob: B*C 概率输出
#         :return:
#         """
#         output_id = output_id.view(-1, 1)  # B*1
#         input_id = input_id.view(-1, 1)  # B*1
#
#         out_prob = prob.gather(1, output_id)  # B*L
#         inp_prob = prob.gather(1, input_id)  # B*L
#
#         # 如何考虑输入汉字和正确汉字之间的概率差值
#         # 正确汉字汉字的概率越大越好，并且一定要大于等于输入汉字
#         # 负log似然已经考虑了这个问题
#
#         zero_t = torch.zeros_like(inp_prob)
#         out_loss_prob = self.margin - out_prob + inp_prob
#         loss = torch.max(zero_t, out_loss_prob)
#
#         # loss = - torch.log(out_prob)
#
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


class CombineLoss(nn.Module):
    """
    nll 负对数似然
    可用于处理多个类别之间的不平衡问题，对某个类别做处理
    这里我希望，更加关注输入输出也就是错误位置的loss
    """

    def __init__(self, p=1, size_average=True):
        super(CombineLoss, self).__init__()
        self.size_average = size_average
        self.p = p
        # self.ignore_index = 0
        # ignore_index=self.ignore_index,
        self.fct = nn.LogSoftmax(dim=-1)
        self.criterion_c = nn.NLLLoss(reduction="none")

        # loss_fct = CrossEntropyLoss()  # -100 index = padding token
        # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.contiguous().view(-1))

    def forward(self, logits, input_id, output_id):
        """
        :param input_id: B
        :param output_id: B
        :param model_logprob: B*C 概率输出
        :return:
        """
        # output_id = output_id.view(-1, 1)  # B*1
        # input_id = input_id.view(-1, 1)  # B*1
        prob = self.fct(logits)

        correct_mask = (input_id == output_id)
        error_mask = (input_id != output_id)
        # 更加关注有错误的位置
        b, s, v = prob.size()

        # self.criterion_c = nn.MultiMarginLoss(p=1, margin=1)
        # MarginLoss()
        # CrossEntropyLoss
        # self.criterion_focal = FocalLoss(gamma=2)
        # ignore_index=0

        loss = correct_mask * self.criterion_c(prob.view(b * s, v), output_id.contiguous().view(-1)).view(b, s) \
               + self.p * error_mask * self.criterion_c(prob.view(b * s, v), output_id.contiguous().view(-1)).view(b, s)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
