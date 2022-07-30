# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 01:11:57 2022

@author: nightrain

an example for loading training by pytorch

"""
#%%
"""
PyTorch provides two data primitives: 

torch.utils.data.DataLoader and torch.utils.data.Dataset 

that allow you to use pre-loaded datasets as well as your own data. 

Dataset stores the samples and their corresponding labels, and 

DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
"""
#%%
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
#%%
train_csv_name = "fashion-mnist_train.csv"
test_csv_name = "fashion-mnist_test.csv"
img_dir = "D:\\Python\\datasets\\archive"


csv_name=train_csv_name
label_name = "label"

#%%
img_filename = csv_name
img_dir = img_dir
label_name = "label"

#%%

img_path = os.path.join(img_dir,img_filename)
img_df = pd.read_csv(img_path)
#%%
# iloc 用index數字取代文字
img_df.iloc[0][1:4].values
#%%
# Extracting all the other columns except label_name
img_cols = [ i for i in img_df.columns if i not in label_name]
image=img_df.iloc[0][img_cols].values
label=int(img_df.iloc[0][0])

image = image.reshape(28,28)
image = image.astype(float)

#藉此得到向量化的圖片資料

#%%
"""
Data does not always come in its final processed form 
that is required for training machine learning algorithms. 

We use transforms to perform some manipulation of the data and 
make it suitable for training.

All TorchVision datasets have two parameters 

transform to modify the features and 
target_transform to modify the labels
"""
#%%
from torchvision.transforms import ToTensor , Lambda
#%%
transforms = ToTensor()
"""
ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. 
and scales the image’s pixel intensity values in the range [0., 1.]

"""
transforms(image)
#%%
"""
Lambda Transforms
Lambda transforms apply any user-defined lambda function. 

Here, we define a function to turn the integer into a one-hot encoded tensor. 

It first creates a zero tensor of size 10 (the number of labels in our dataset) and 

calls scatter_ which assigns a value=1 on the index as given by the label y.
"""

# Establishing labels externally 
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# Converting y-labels to one hot encoding
target_transform = Lambda(lambda y: torch.zeros(
    len(labels_map.items()), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
#%%
target_transform(label)

#%%
## Creating a custom dataset for your files

class CustomFashionMNISTCsvDataset(Dataset):
    def __init__(self, csv_name, img_dir, transform=None, target_transform=None , label_name = "label"):
        
        self.img_filename = csv_name
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_name = label_name
        
        img_path = os.path.join(self.img_dir, self.img_filename)
        self.img_df = pd.read_csv(img_path)

    def __len__(self):
        # the length of datasets
        return len(self.img_df)

    def __getitem__(self, idx):
        # get idx-th sample data (image, label)
        
        # Extracting all the other columns except label_name
        img_cols = [ i for i in self.img_df.columns if i not in self.label_name]
        
        image = self.img_df.iloc[[idx]][img_cols].values
        label = int(self.img_df.iloc[[idx]][self.label_name])
        
        # Reshaping the array from 1*784 to 28*28
        
        image = image.reshape(28,28)
        image = image.astype(float)
        
        # Scaling the image so that the values only range between 0 and 1
        # image = image/255.0
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        image = image.to(torch.float)
        label = label.to(torch.float)
        
        return image, label
#%%
temp_train_dataset = CustomFashionMNISTCsvDataset(csv_name = train_csv_name ,
                                                  img_dir = img_dir , 
                                                  transform = transforms , 
                                                  target_transform = target_transform , 
                                                  label_name = label_name)
#%%
sample_img , sample_lbl =temp_train_dataset[1]
len(temp_train_dataset)
#%%
import matplotlib.pyplot as plt

plt.imshow(sample_img.squeeze(), cmap="gray")

sample_lbl
#%%

# data loader  select a part of datasets
# Data loader. Combines a dataset and a sampler, 
# and provides an iterable over the given dataset.

from torch.utils.data import DataLoader

temp_train_dataloader = DataLoader(temp_train_dataset, batch_size=64, shuffle=True)
temp_test_dataloader = DataLoader(temp_train_dataset, batch_size=64, shuffle=True)
#%%
train_features, train_labels = next(iter(temp_train_dataloader))

#The iter() function creates an object which can be iterated one element at a time.

#The next() function returns the next item from the iterator.

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
#%%

