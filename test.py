import os
import torch
import torch.nn as nn
from torch.autograd import Variable

rnn1 = nn.ConvLSTM(3, 10, 2, kernel_size=3)
input1 = Variable(torch.randn(4, 10, 3, 25, 25))
h01 = Variable(torch.randn(2, 10, 10, 25, 25))
c01 = Variable(torch.randn(2, 10, 10, 25, 25))
output1, hn1 = rnn1(input1) #, (h01, c01)
print('input1:{}'.format(input1.size()))
print('output1:{}'.format(output1.size()))
print('hn1:{}'.format(hn1[0][0].size()))


# input1:torch.Size([4, 10, 3, 25, 25])
# output1:torch.Size([4, 10, 10, 25, 25])
# hn1:torch.Size([10, 10, 25, 25])