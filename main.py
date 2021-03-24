import torch
from torch import nn
import torch.nn.functional as F
from Model import Perception
from torch import optim
from tensorboardX import SummaryWriter
import pandas as pd

data = pd.read_csv("D:\city_temperature.csv")
list=data.loc[:3000, ['AvgTemperature']].astype('float').values
del data
input=[]
for i in range(10):
    input.append(list[i][0])
writer=SummaryWriter('logs/wendu')
model=Perception(10,3,1)
optimizer=optim.SGD(params=model.parameters(),lr=0.02)
criterion = nn.CrossEntropyLoss()

for i in range(1000):
    input.pop(0)
    input.append(list[i+10][0])
    input=torch.Tensor(input).float()
    label=list[i+11]
    label=torch.Tensor(label).float()
    output=model(input)
    loss=criterion(output,label)
    writer.add_scalar('loss/total_loss',loss.item(),i)
    if i%50==0:
        print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

