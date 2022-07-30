# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 19:11:00 2022

@author: nightrain
"""
import torch
import torch.nn as nn
import torch.optim as optim
from ModelClass import MyCNN
import matplotlib.pyplot as plt
#%%
model_CNN = MyCNN()
print(model_CNN)
#%%
print(model_CNN.count_parameters())
#%%
pic = torch.randn(1, 1, 32, 32)
out = model_CNN(pic)
print(out)
#%%
out = model_CNN(pic)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss)
#%%
model_CNN.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(model_CNN.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(model_CNN.conv1.bias.grad)
#%%
"""
torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). 
               Also holds the gradient w.r.t. the tensor.

nn.Module - Neural network module. Convenient way of encapsulating parameters, 
            with helpers for moving them to GPU, exporting, loading, etc.

nn.Parameter - A kind of Tensor, that is automatically registered as a parameter 
               when assigned as an attribute to a Module.

autograd.Function - Implements forward and backward definitions of an autograd operation. 
                    Every Tensor operation creates at least a single Function node 
                    that connects to functions that created a Tensor and encodes its history.
"""
#%%
lossVal=[]
# create your optimizer
optimizer = optim.SGD(model_CNN.parameters(), lr=0.01)
for i in range(100):
    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    out = model_CNN(pic)
    loss = criterion(out, target)
    lossVal.append(loss.detach().numpy())
    
    print('conv1.bias before update')
    print(model_CNN.conv1.bias)
    
    loss.backward()
    optimizer.step()    # Does the update
    
    print('conv1.bias after update')
    print(model_CNN.conv1.bias)
#%%
plt.plot(lossVal)