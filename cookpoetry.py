#模式是一个词接一个词的预测
import os
import re
import json

import torch.nn as nn
import torch
from torch.autograd import Variable
from collections import deque
import random
import pickle as p
import time

def process_unexpected_symbol(paragrahs):
    paragrahs_string=""
    for senntence in paragrahs:
        paragrahs_string=paragrahs_string+senntence
    out,number=re.subn("（.*）","",paragrahs_string)
    out, number = re.subn("（.*）", "",out)
    out, number = re.subn("-.*-", "", out)
    out, number = re.subn("《.*》", "", out)
    out, number = re.subn("[\[\]]", "", out)
    out, number = re.subn("{}", "", out)
    return out
def cookdata():
    path = "poetry/"
    filelist = os.listdir(path)
    filepath = []
    data = []
    for i in filelist:
        if ("tang" in i) & ("authors" not in i):
            file=path+i
            filepath.append(file)
    for file in filepath:
        poetrylist=json.load(open(file,'rb'))
        for poetrydic in poetrylist:
            process_result=process_unexpected_symbol(poetrydic["paragraphs"])
            if "{" not in process_result:
                data.append(process_result)
    return data

def  construct_dic():
    data=cookdata()
    word_to_dic={}
    for poetry in data:
        for word in poetry:
            if word not in word_to_dic:
                word_to_dic[word]=len(word_to_dic)
    word_to_dic["<EOP>"]=len(word_to_dic)
    word_to_dic["<START>"]=len(word_to_dic)
    return word_to_dic,data


def makesample(s, tensor_dic):
    input = []
    output = []
    # IN = ""
    # OUT = ""
    for i in range(1, len(s)):
        # IN = IN + s[i - 1]
        # OUT = OUT + s[i]
        input.append(tensor_dic[s[i - 1]])
        output.append(tensor_dic[s[i]])
    return torch.cat(input), torch.cat(output)

def data2datalist(data):
    datalist=[]
    list=[]
    for i in range(0,len(data)):
        for word in data[i]:
            list.append(word)
        list.append("<EOP>")
        datalist.append(list)
        list=[]
    return  datalist


class poetry_model(nn.Module):
    #我们是在词典中采取，多分类方式进行选择的
    def __init__(self,dic_len,embed_dim,hidden_dim):
        super(poetry_model, self).__init__()
        self.embedding=nn.Embedding(dic_len,embed_dim)
        self.lstm=nn.LSTM(input_size=embed_dim,hidden_size=hidden_dim,num_layers=2,batch_first=True)
        self.linear=nn.Sequential(nn.Linear(hidden_dim,dic_len),
                                  nn.ReLU(),
                                  nn.LogSoftmax(),)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_ih_l1)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l1)
    def forward(self,x):
        seq=x.shape[0]
        out=self.embedding(x)#Because I set batch_first=True,so input (batch,seq,feature)
        out=out.view((1,seq,256))
        out,(h_N,c_N)=self.lstm(out)
        out=out.view(seq,-1)
        out=self.linear(out)
        return out



word_to_dic,data=construct_dic()
file=open('word_Dic.pkl', 'wb')
p.dump(word_to_dic,file)
tensor_dic={}
data_deque=deque(maxlen=1000000)

for word in word_to_dic:
    tensor_dic.setdefault(word,Variable(torch.LongTensor([word_to_dic[word]])))

datalist=data2datalist(data)

dic_len=len(word_to_dic)
model=poetry_model(dic_len,256,256)
for poetry in datalist:
    if len(poetry)>2:
        data_deque.append(poetry)#deque抽样速度更快,去掉长度为1，或2

# input,output=makesample(datalist[0],tensor_dic)
# result=model(input)
# print(result)

# l=[]
# for lp in datalist:
#     l.append(len(lp))
# l.sort()
# print(l)


#train


loss= nn.NLLLoss()
minibatch=64
optimizer=torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)



for epoch in range(1):
     for sample_to_train in range(int(len(data_deque)/minibatch)):
         samples=random.sample(data_deque,minibatch)
         Loss = 0
         for sample in samples:
             input, output_lable = makesample(sample, tensor_dic)
             output=model(input)
             Loss+=loss(output,input)
         optimizer.zero_grad()
         Loss = Loss / 64
         Loss.backward()
         print(epoch, Loss.data[0])
         optimizer.step()
torch.save(model, 'cook_poetry.pt')




















