# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:54:36 2022

@author: nightrain
"""

# %% [code] 
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('D:\\Python\\datasets\\archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## Creating a custom dataset wrapper 
# to read the images from csv and display them as images

# %% [code] 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor , Lambda

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")


print(f"Using {device} device")

dtype = torch.FloatTensor
# cudnn.benchmark = True
#%%
## Creating a custom dataset for your files

import os
import pandas as pd
from torchvision.io import read_image

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
        return len(self.img_df)

    def __getitem__(self, idx):
        
        # Extracting all the other columns except label_name
        img_cols = [ i for i in self.img_df.columns if i not in self.label_name]
        
        image = self.img_df.iloc[[idx]][img_cols].values
        label = int(self.img_df.iloc[[idx]][self.label_name].values)
        
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


# %% [markdown]
# # Creating datasets and loaders for finding mean and std for normalization

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:33:30.968257Z","iopub.execute_input":"2022-06-26T18:33:30.968804Z","iopub.status.idle":"2022-06-26T18:33:38.906099Z","shell.execute_reply.started":"2022-06-26T18:33:30.968767Z","shell.execute_reply":"2022-06-26T18:33:38.904715Z"}}

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


# Crerating a temp dataset
train_csv_name = "fashion-mnist_train.csv"
test_csv_name = "fashion-mnist_test.csv"
img_dir = "D:\\Python\\datasets\\archive"

# Converting X variables to Tensors
transforms = ToTensor()

# Converting y-labels to one hot encoding
target_transform = Lambda(lambda y: torch.zeros(
    len(labels_map.items()), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

label_name = "label"

temp_train_dataset = CustomFashionMNISTCsvDataset(csv_name = train_csv_name , img_dir = img_dir , transform = transforms , target_transform = target_transform , label_name = label_name)
temp_test_dataset = CustomFashionMNISTCsvDataset(csv_name = test_csv_name , img_dir = img_dir , transform = transforms , target_transform = target_transform , label_name = label_name)
x0 , y0 = temp_train_dataset[0]
print(x0.shape , y0.shape)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:33:38.907767Z","iopub.execute_input":"2022-06-26T18:33:38.908562Z","iopub.status.idle":"2022-06-26T18:33:39.435423Z","shell.execute_reply.started":"2022-06-26T18:33:38.908519Z","shell.execute_reply":"2022-06-26T18:33:39.434358Z"}}
# Ploting some of the datapoints in the dataset
import matplotlib.pyplot as plt

# sample_img , sample_lbl = temp_train_dataset[3]
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
figure.add_subplot(rows, cols, 1)
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(temp_train_dataset), size=(1,)).item()
    sample_img , sample_lbl = temp_train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[torch.argmax(sample_lbl).item()])
    plt.axis("off")
    plt.imshow(sample_img.squeeze(), cmap="gray")
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:33:39.438667Z","iopub.execute_input":"2022-06-26T18:33:39.439048Z","iopub.status.idle":"2022-06-26T18:33:39.444304Z","shell.execute_reply.started":"2022-06-26T18:33:39.439001Z","shell.execute_reply":"2022-06-26T18:33:39.443272Z"}}
# Creating data loaders using temp_dataset
from torch.utils.data import DataLoader

temp_train_dataloader = DataLoader(temp_train_dataset, batch_size=64, shuffle=True)
temp_test_dataloader = DataLoader(temp_train_dataset, batch_size=64, shuffle=True)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:33:39.445934Z","iopub.execute_input":"2022-06-26T18:33:39.446307Z","iopub.status.idle":"2022-06-26T18:33:39.727294Z","shell.execute_reply.started":"2022-06-26T18:33:39.446268Z","shell.execute_reply":"2022-06-26T18:33:39.726204Z"}}
# Ploting image and label using loader class .
train_features, train_labels = next(iter(temp_train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {labels_map[torch.argmax(label).item()]}")

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:33:39.728911Z","iopub.execute_input":"2022-06-26T18:33:39.72926Z","iopub.status.idle":"2022-06-26T18:35:08.959773Z","shell.execute_reply.started":"2022-06-26T18:33:39.72923Z","shell.execute_reply":"2022-06-26T18:35:08.958396Z"}}
# Loading the whole dataset into a single batch to find the mean and standard deviation for normalization
temp_loader = DataLoader(temp_train_dataset, batch_size=int(0.01*len(temp_train_dataset)))
data = next(iter(temp_loader))

mean = data[0].mean() 
std_dev = data[0].std()

# Finding mean and std for normalization of the data
mean, std_dev

# %% [markdown]
# # Implementing Normalization layer using the obtained mean and standard deviation 

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:08.961469Z","iopub.execute_input":"2022-06-26T18:35:08.962207Z","iopub.status.idle":"2022-06-26T18:35:14.070338Z","shell.execute_reply.started":"2022-06-26T18:35:08.962164Z","shell.execute_reply":"2022-06-26T18:35:14.069167Z"}}
from torchvision import  transforms
# Crerating a temp dataset
train_csv_name = "fashion-mnist_train.csv"
test_csv_name = "fashion-mnist_test.csv"
img_dir = "C:\\Users\\nightrain\\Desktop\\archive"

# Converting X variables to Tensors
transform_with_normalization  = transforms.Compose([
          transforms.ToTensor() , transforms.Normalize(mean, std_dev)
    ])

# Converting y-labels to one hot encoding
target_transform = Lambda(lambda y: torch.zeros(
    len(labels_map.items()), dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

label_name = "label"

train_dataset = CustomFashionMNISTCsvDataset(csv_name = train_csv_name , img_dir = img_dir , transform = transform_with_normalization , target_transform = target_transform , label_name = label_name)

## Applying the same tranformation to the test data
## We keep the mean and std as same since we consider test and train data to have the same distribution 

test_dataset = CustomFashionMNISTCsvDataset(csv_name = test_csv_name , img_dir = img_dir , transform = transform_with_normalization , target_transform = target_transform , label_name = label_name)

x0 , y0 = train_dataset[0]
print(x0.shape , y0.shape)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:14.071962Z","iopub.execute_input":"2022-06-26T18:35:14.072596Z","iopub.status.idle":"2022-06-26T18:35:14.085557Z","shell.execute_reply.started":"2022-06-26T18:35:14.072554Z","shell.execute_reply":"2022-06-26T18:35:14.084577Z"}}
print(x0)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:14.08723Z","iopub.execute_input":"2022-06-26T18:35:14.087883Z","iopub.status.idle":"2022-06-26T18:35:14.09428Z","shell.execute_reply.started":"2022-06-26T18:35:14.087845Z","shell.execute_reply":"2022-06-26T18:35:14.093321Z"}}
# Creating data loaders using temp_dataset
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# %% [markdown]
# ## Model creation 

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:14.098645Z","iopub.execute_input":"2022-06-26T18:35:14.099051Z","iopub.status.idle":"2022-06-26T18:35:17.067993Z","shell.execute_reply.started":"2022-06-26T18:35:14.099015Z","shell.execute_reply":"2022-06-26T18:35:17.067033Z"}}
from torch import nn

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class MyOwnNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyOwnNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return torch.softmax(logits , dim = 1)
    
model = MyOwnNeuralNetwork().to(device)
print(model)

#model = model.to(device)

#torch.backends.cudnn.benchmark=True
#torch.cuda.set_device(0)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:17.069537Z","iopub.execute_input":"2022-06-26T18:35:17.069916Z","iopub.status.idle":"2022-06-26T18:35:17.079516Z","shell.execute_reply.started":"2022-06-26T18:35:17.06987Z","shell.execute_reply":"2022-06-26T18:35:17.077352Z"}}
## Softmax example 

import torch
a = torch.tensor([0.2,0.3,0.5,0.8])
torch.softmax(a , dim = 0)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:17.081066Z","iopub.execute_input":"2022-06-26T18:35:17.083127Z","iopub.status.idle":"2022-06-26T18:35:17.090043Z","shell.execute_reply.started":"2022-06-26T18:35:17.083082Z","shell.execute_reply":"2022-06-26T18:35:17.088945Z"}}
## Defining optimizer and loss functions 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:36:16.084992Z","iopub.execute_input":"2022-06-26T18:36:16.085633Z","iopub.status.idle":"2022-06-26T18:36:16.103883Z","shell.execute_reply.started":"2022-06-26T18:36:16.085585Z","shell.execute_reply":"2022-06-26T18:36:16.10175Z"}}
from torch.autograd import Variable
def train(dataloader, model, loss_fn, optimizer):
    
    # Total size of dataset for reference
    size = len(dataloader.dataset)
    
    # places your model into training mode
    model.train()
    
    
    # Gives X , y for each batch
    for batch, (X, y) in enumerate(dataloader):
        
        # X, y = X.to(device), y.to(device)
        X, y = X, y#.cuda()
        #model.cuda()
        # dtype = torch.cuda.FloatTensor
        # X = Variable(X).type(dtype)
        # y = Variable(y).type(dtype)

        # Compute prediction error / loss
        # 1. Compute y_pred 
        # 2. Compute loss between y and y_pred using selectd loss function
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0 
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:27.413866Z","iopub.execute_input":"2022-06-26T18:35:27.414233Z","iopub.status.idle":"2022-06-26T18:35:27.577535Z","shell.execute_reply.started":"2022-06-26T18:35:27.414203Z","shell.execute_reply":"2022-06-26T18:35:27.57436Z"}}
data = next(iter(train_dataloader))
data[0].shape , data[1].shape


# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:29.32033Z","iopub.execute_input":"2022-06-26T18:35:29.321541Z","iopub.status.idle":"2022-06-26T18:35:29.330748Z","shell.execute_reply.started":"2022-06-26T18:35:29.321485Z","shell.execute_reply":"2022-06-26T18:35:29.32966Z"}}
def test(dataloader, model, loss_fn):
    
    # Total size of dataset for reference
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Explanation given above
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        
        # Gives X , y for each batch
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            model.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    
    ## Calculating loss based on loss function defined
    test_loss /= num_batches
    
    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:30.417117Z","iopub.execute_input":"2022-06-26T18:35:30.417522Z","iopub.status.idle":"2022-06-26T18:35:30.423294Z","shell.execute_reply.started":"2022-06-26T18:35:30.417488Z","shell.execute_reply":"2022-06-26T18:35:30.421981Z"}}
print(device)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:31.283934Z","iopub.execute_input":"2022-06-26T18:35:31.28468Z","iopub.status.idle":"2022-06-26T18:35:31.289783Z","shell.execute_reply.started":"2022-06-26T18:35:31.28464Z","shell.execute_reply":"2022-06-26T18:35:31.288378Z"}}
cuda_ = "cuda:0"
device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:32.070228Z","iopub.execute_input":"2022-06-26T18:35:32.070613Z","iopub.status.idle":"2022-06-26T18:35:32.078798Z","shell.execute_reply.started":"2022-06-26T18:35:32.070577Z","shell.execute_reply":"2022-06-26T18:35:32.077789Z"}}
model.to(device)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:35:34.476324Z","iopub.execute_input":"2022-06-26T18:35:34.477245Z","iopub.status.idle":"2022-06-26T18:35:34.484231Z","shell.execute_reply.started":"2022-06-26T18:35:34.477204Z","shell.execute_reply":"2022-06-26T18:35:34.483032Z"}}
#torch.cuda.get_device_name()

# %% [code] {"execution":{"iopub.status.busy":"2022-06-26T18:36:21.820946Z","iopub.execute_input":"2022-06-26T18:36:21.821292Z","iopub.status.idle":"2022-06-26T18:40:04.838976Z","shell.execute_reply.started":"2022-06-26T18:36:21.82126Z","shell.execute_reply":"2022-06-26T18:40:04.837983Z"}}
USE_GPU = False
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# %% [code]
# cuda_ = "cuda:0"
# device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
# model.to(device)


# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")


# %% [code]
