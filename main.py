import torch
from torch import nn
import torch.nn.functional as F
from Model import Perception
from torch import optim
from tensorboardX import SummaryWriter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data = pd.read_csv("D:\city_temperature.csv")[:3000]
# a=np.array(data['AvgTemperature'])
# plt.hist(a, bins =  range(100))
# plt.title("histogram")
# plt.show()
# plt.plot(range(3000),data['AvgTemperature'])
# plt.show()
# max=np.array([data.max().values[7]])
# min=np.array([data.min().values[7]])
max=np.array([84])
min=np.array([43])
list=data.loc[:3000, 'AvgTemperature'].astype('float').values
for i in range(3000):
    if(list[i]>84):list[i]=84
    if(list[i]<43):list[i]=43
list=(list-min)/(max-min)
del data
input=[]
for i in range(1000):
    input.append(list[i:i+10])
label=list[10:1010]
writer=SummaryWriter('logs/wendu')
model=Perception(10,100,20,3,1)
optimizer=optim.SGD(params=model.parameters(),lr=2)
MSE = nn.MSELoss()

for i in range(2000):
    input=torch.tensor(input).float()
    label=torch.tensor(label).reshape([1000,1]).float()
    output=model(input)
    loss=MSE(output,label)
    writer.add_scalar('loss/total_loss',loss.item(),i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%50==0:
        print(loss.item())
    if i==1999:
        label.reshape([1000])
        output.reshape([1000])
        label=label.detach().numpy()
        output=output.detach().numpy()
        x=range(1000)
        plt.plot(x,label,"g")
        plt.plot(x,output,"r")
        plt.show()

test_x=[]
for i in range(1000,1100):
    test_x.append(list[i:i+10])
test_label=list[1010:1110]
test_x=torch.tensor(test_x).float()
test_y=model(test_x)
test_y.reshape([100])
test_y=test_y.detach().numpy()
x=range(100)
plt.plot(x,test_label,"g")
plt.plot(x,test_y,"r")
plt.show()