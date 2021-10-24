
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import scipy.io as sio


###Q K V
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v):
        # q x k^T
        # attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  ### H*W  * W*H
        # # dim=-1表示对最后一维softmax
        # attn = self.dropout(F.softmax(attn, dim=-1))
        # output = torch.matmul(attn, v)  #  H*H  *  H*W  得到H个权重，这和我的时间片加权不符合，加到了30通道上，所以改成下面的
        attn = torch.matmul(q.transpose(2, 3) / self.temperature, k)  ### W*H  * H*W
        # dim=-1表示对最后一维softmax
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(v,attn)  #   H*W * W*W  得到W个权重
        return output

####### 里面多尺度卷积做过消融实验，可能乱了
##used
class Tblock3(nn.Module):   ###256采样
    def __init__(self,  input_size, num_T):
        # input_size: EEG channel x datapoint
        super(Tblock3, self).__init__()  # 子类把父类的__init__()放到自己的__init__()当中

        self.Tception = nn.Sequential(
            nn.Conv2d(input_size[0], input_size[0], kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(), )
        size = self.get_size(input_size)
        self.attention = ScaledDotProductAttention(temperature=size[3] ** 0.5)
        #self.layer_norm = nn.LayerNorm(size[3], eps=1e-6)

        self.Tception1 = nn.Sequential(  #63--->28
            nn.Conv2d(input_size[0], num_T, kernel_size=(1, 7), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))

        self.Tception3 = nn.Sequential(   #63--->30
            nn.Conv2d(input_size[0], num_T, kernel_size=(1, 128), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))

        self.Tception4 = nn.Sequential(
            nn.Conv2d(input_size[0], num_T, kernel_size=(1, 400), stride=1, padding=0),
            nn.ReLU(),
            )


        self.BN_t = nn.BatchNorm2d(num_T)  # 进行数据的归一化处理

    #  1*num_T*30*t
    def forward(self, x):

        input = self.Tception(x)
        q,k,v=input,input,input
        out=self.attention(q,k,v)
        input = out+input
        #input = self.layer_norm(input)

        #########

        y = self.Tception1(input)
        out = y

        y = self.Tception3(input)
        out = torch.cat((out, y), dim=3)  # 行连接

        y = self.Tception4(input)
        out = torch.cat((out, y), dim=3)


        out = self.BN_t(out)

        return out

    def get_size(self, input_size):  ##加权个数
        data = torch.ones((1, input_size[0], input_size[1], input_size[2]))
        y = self.Tception(data)
        out = y
        #print(out.size())
        return out.size()

class Tblock4(nn.Module):   ###256采样
    def __init__(self,  input_size, num_T):
        # input_size: EEG channel x datapoint
        super(Tblock4, self).__init__()  # 子类把父类的__init__()放到自己的__init__()当中

        self.Tception = nn.Sequential(
            nn.Conv2d(input_size[0], input_size[0], kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(), )
        size = self.get_size(input_size)
        self.attention = ScaledDotProductAttention(temperature=size[3] ** 0.5)
        #self.layer_norm = nn.LayerNorm(size[3], eps=1e-6)

        self.Tception1 = nn.Sequential(  #63--->28
            nn.Conv2d(input_size[0], num_T, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))


        self.BN_t = nn.BatchNorm2d(num_T)  # 进行数据的归一化处理

    #  1*num_T*30*t
    def forward(self, x):

        input = self.Tception(x)
        q,k,v=input,input,input
        out=self.attention(q,k,v)
        input = out+input
        #input = self.layer_norm(input)

        #########

        y = self.Tception1(input)
        out = y
        out = self.BN_t(out)
        return out

    def get_size(self, input_size):  ##加权个数
        data = torch.ones((1, input_size[0], input_size[1], input_size[2]))
        y = self.Tception(data)
        out = y
        #print(out.size())
        return out.size()
class TSCN(nn.Module):

    def __init__(self, num_classes, input_size, num_T, num_S,
                 hiden,dropout_rate):

        super(TSCN, self).__init__()

        self.Get_timefeatures = Tblock3(input_size=input_size, num_T=num_T, )  ###为了得到时间特征的个数 Tsize

        self.BN_s = nn.BatchNorm2d(num_S)

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)))


        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, num_classes),
            # nn.Softmax())
            nn.LogSoftmax())


    def forward(self,X ):

        out = self.Get_timefeatures(X)  ### B*9*30*Tfetures
        z = self.Sception1(out)
        out_final = z

        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        out= self.fc1(out)
        out=self.fc2(out)
        return out


    def get_size(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], input_size[2]))
        out=self.Get_timefeatures(data)
        z = self.Sception1(out)
        out_final = z

        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)

        return out.size()


class TSCN2(nn.Module):

    def __init__(self, num_classes, input_size, num_T, num_S,
                 hiden,dropout_rate,hiden2,dropout_rate2):

        super(TSCN2, self).__init__()

        self.Get_timefeatures = Tblock4(input_size=input_size, num_T=num_T, )  ###为了得到时间特征的个数 Tsize

        self.BN_s = nn.BatchNorm2d(num_S)

        self.Sception1 = nn.Sequential(
            nn.Conv2d(num_T, num_S, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1)))


        size = self.get_size(input_size)

        self.fc1 = nn.Sequential(
            nn.Linear(size[1], hiden),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        self.fc2 = nn.Sequential(
            nn.Linear(hiden, hiden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate2))
        self.fc3 = nn.Sequential(
            nn.Linear(hiden2, num_classes),
            nn.LogSoftmax())


    def forward(self,X ):

        out = self.Get_timefeatures(X)  ### B*9*30*Tfetures
        z = self.Sception1(out)
        out_final = z

        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        out= self.fc1(out)
        out=self.fc2(out)
        return out


    def get_size(self, input_size):
        data = torch.ones((1, input_size[0], input_size[1], input_size[2]))
        out=self.Get_timefeatures(data)
        z = self.Sception1(out)
        out_final = z

        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)

        return out.size()

if __name__ == "__main__":
    # model = DGCNN(3, (1, 30, 32), 9, 6,20,16,12,128,0.2)
    # model = AMCNNDGCN(3, (1, 30, 256), 9, 6, 20, 16, 12, 128, 0.2)
    #model = TGCN_withGCNlayer2(3, (1, 30, 256), 9, 6, 128,   43,    128, 0.2)
    model = TSCN(2, (1, 1, 729), 12, 18, 128, 0.2)
    print(model)

