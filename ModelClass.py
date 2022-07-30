# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 22:22:46 2022

@author: nightrain

we make a note for oop network model

"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%

"""
The __init__ method is the Python equivalent of 
the C++ constructor in an object-oriented approach.

The __init__  function is called every time 
an object is created from a class. 

The __init__ method lets the class initialize the 
object’s attributes and serves no other purpose. 
It is only used within classes. 
"""
#%%
"""
The super(Net, self).__init__() refers to the fact that 
this is a subclass of nn.Module and is inheriting all methods.
In the super class, nn.Module, 
there is a __call__ method which obtains the forward function 
from the subclass and calls it.
"""
#%%
"""
繼承nn.Model造自己的model
super 用來直接call forword function data帶入model就會計算
enumerate() : It allows us to loop over something and have an automatic counter
"""
#%%
# myDNN model
class DNN(nn.Module):
    def __init__(self,InDim,OutDim,depth,neurons,Addbias=True):
        super(DNN,self).__init__()
        self.InputLayer=nn.Linear(InDim, neurons,bias=Addbias)
        self.HiddenLayers = \
            nn.ModuleList([nn.Linear(neurons, neurons,bias=Addbias) for i in range(depth-1)])
        self.OutputLayer=nn.Linear(neurons,OutDim,bias=Addbias)
        self.activation=nn.Tanh()
        
    def forward(self,data):
        output=self.InputLayer(data)
        for l in (self.HiddenLayers):
            output = l(self.activation(output))
        output=self.OutputLayer(self.activation(output))
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
#%%

#simple code check

# #InDim, OutDim, depth, neurons
# model=DNN(2,3,2,50,False)
# model
# #%%  
# model(2*torch.randn(10,2))
# #%%
# model.count_parameters()
#%%
# define a CNN model
class MyCNN(nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, data):
        # Max pooling over a (2, 2) window
        output = F.max_pool2d(F.relu(self.conv1(data)), (2, 2))
        # If the size is a square, you can specify with a single number
        output = F.max_pool2d(F.relu(self.conv2(output)), 2)
        output = torch.flatten(output, 1) # flatten all dimensions except the batch dimension
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


#%%
        