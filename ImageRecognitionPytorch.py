# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:49:40 2022

@author: nightrain
"""

from torchvision import models
dir(models)
#%%
resnet = models.resnet101(pretrained=True)
#%%
"""
we need to preprocess the input images so that they are the correct size 
and their values ​​(colours) are roughly in the same numeric range.

provides transformations, 
which will allow us to quickly 
define pipelines of basic preprocessing functions
"""
#%%
from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
"""
preprocessing function that will scale the input image to 256 × 256, 
crop the image to 224 × 224 around the centre, turn it into a tensor, 
and normalize its RGB components.
"""
#%%
path='D:\\Python\\datasets\\ImageRecPytorch\\dog.png'
#%%
from PIL import Image
img = Image.open(path)
#%%
img_t=preprocess(img)
#%%
import torch
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
out
#%%








