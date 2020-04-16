# # coding=utf-8
#
# import numpy as np
# import matplotlib.pyplot as plt  # 导入模块
#
# coding=utf-8
from __future__ import absolute_import, print_function
import torch.utils.data
from torch.backends import cudnn

cudnn.benchmark = True

orth_score_list = list()
for i in range(100, 1200, 100):
    resume_path = 'checkpoints/knnsoftmax/%d_model.pkl' % i
    model = torch.load(resume_path)
    model_dict = model.state_dict()
    w = model_dict['Embed.linear.weight']
    orth_mat = torch.matmul(w, w.t())
    mean = torch.mean(torch.masked_select(orth_mat, torch.eye(w.size()[0]).cuda() == 1))
    orth_mat = orth_mat / mean
    orth_mat = orth_mat - torch.eye(w.size()[0]).cuda()
    orth_score = torch.mean(torch.abs(orth_mat))
    orth_score_list.append(orth_score)
# #
# orth_score = [0, 0.008148077875375748,
#              0.010764598846435547,
#              0.012241244316101074,
#                  0.012818210758268833,
#                  0.013539150357246399,
#                  0.014230653643608093,
#                  0.01473081111907959,
#                  0.016075268387794495,
#                  0.017009025439620018,
#                  0.018675770610570908,
#                  0.019362201914191246]
#
#
# epoch = range(0, 1200, 100)
#
# plt.plot(epoch, orth_score)
# plt.xlabel('Epoch')
# plt.ylabel('Distance to orth-mat')
# plt.savefig('orth.jpg')
#
# plt.show()  # 输出图像
