import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.image as mpimg
import plotly.express as px
from sklearn.manifold import TSNE
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import decomposition

import os

# %% [markdown]
# # Statistic Description

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:17:18.830398Z","iopub.execute_input":"2022-06-06T15:17:18.831207Z","iopub.status.idle":"2022-06-06T15:17:18.989813Z","shell.execute_reply.started":"2022-06-06T15:17:18.831174Z","shell.execute_reply":"2022-06-06T15:17:18.987821Z"}}
birdSpecies = pd.read_csv('D:/Python/datasets/bird/birds.csv')
birdSpecies.info()

# %% [code] {"execution":{"iopub.status.busy":"2022-06-03T00:53:13.155973Z","iopub.execute_input":"2022-06-03T00:53:13.156885Z","iopub.status.idle":"2022-06-03T00:53:13.173686Z","shell.execute_reply.started":"2022-06-03T00:53:13.156819Z","shell.execute_reply":"2022-06-03T00:53:13.172697Z"}}
birdSpecies.head(5)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:22:20.851158Z","iopub.execute_input":"2022-06-02T16:22:20.851537Z","iopub.status.idle":"2022-06-02T16:22:20.872233Z","shell.execute_reply.started":"2022-06-02T16:22:20.851499Z","shell.execute_reply":"2022-06-02T16:22:20.87113Z"}}
result = birdSpecies.groupby(['labels']).size()
result.describe()

# %% [markdown]
# * There are 400 different species of birds in the dataset
# * The average size of species is 155 birds

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:22:20.873946Z","iopub.execute_input":"2022-06-02T16:22:20.874419Z","iopub.status.idle":"2022-06-02T16:22:21.159631Z","shell.execute_reply.started":"2022-06-02T16:22:20.874371Z","shell.execute_reply":"2022-06-02T16:22:21.158855Z"}}
sns.set(rc = {'figure.figsize':(10,15)})
ax = sns.boxplot(data=result)
q1 = 139
q3 = 166
outlier_top = q3 + 1.5 * (q3 - q1)
outlier_bottom = q1 - 1.5 * (q3 - q1)

for val in result:
    if val > outlier_top or val < outlier_bottom:
        label = result[result == val].index[0]
        plt.text(x=0, y=val, s=label)

# %% [markdown]
# * There are upper half outlier: HOUSE FINCH, OVENBIRD, D-ARNAUDS BARBET, SWINHOES PHEASANT, WOOD DUCK, CASPIAN TERN, RED TAILED HAWK, MARABOU STORK;
# * The largest size of species contains 259 bird images, which is "HOUSE FINCH"
# * The smallest size of species contains 10 bird images, which is "BLACK & YELLOW BROADBILL"

# # %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:22:21.161165Z","iopub.execute_input":"2022-06-02T16:22:21.161753Z","iopub.status.idle":"2022-06-02T16:22:21.792062Z","shell.execute_reply.started":"2022-06-02T16:22:21.161712Z","shell.execute_reply":"2022-06-02T16:22:21.790498Z"}}
# # size() is equivalent to counting the distinct rows
# result = birdSpecies.groupby(['data set']).size()
# result = result.sort_values(ascending=False)
# text = np.around([result.values[0], result.values[1], result.values[2]]/result.values.sum()*100,2)
# # plot the result
# sns.set(rc = {'figure.figsize':(8,10)})
# fig = px.bar(result, x=result.index, y=result.values)
# #,text=text,animation_frame=(),title="Birds in each data set", labels={"y":"Count"}
# fig.show()

# %% [markdown]
# # Visualize Transformation in Pytorch

# %% [markdown]
# **Random samples from dataset**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:17:24.415659Z","iopub.execute_input":"2022-06-06T15:17:24.416012Z","iopub.status.idle":"2022-06-06T15:17:24.422543Z","shell.execute_reply.started":"2022-06-06T15:17:24.415984Z","shell.execute_reply":"2022-06-06T15:17:24.420736Z"}}
local_path = 'D:/Python/datasets/bird'

# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T15:46:15.700789Z","iopub.execute_input":"2022-06-04T15:46:15.70145Z","iopub.status.idle":"2022-06-04T15:46:17.585143Z","shell.execute_reply.started":"2022-06-04T15:46:15.701414Z","shell.execute_reply":"2022-06-04T15:46:17.584315Z"}}
local_path = 'D:/Python/datasets/bird/'

fig, axs = plt.subplots(ncols=5, nrows=5, figsize=(15, 15))
for i in range(5):
    for j in range(5):
        sample = birdSpecies.sample()
        species = sample['labels'].to_string(index=False)
        index = sample.index
        filepath = str(sample.filepaths.values)[2:-2]
        random_bird = mpimg.imread(local_path + filepath)
        plt.imshow(random_bird)
        axs[i, j].imshow(random_bird)
        axs[i, j].set_title(species)
        axs[i, j].axis('off')

# %% [markdown]
# * In neural network model, data augmentation methos is applied on images to prevent overfitting. All transformation applied after convert images to numpy array.
# * The first row in the plot below shows the random select original image in the dataset, the 2nd row shows the result after normalize the numpy array of images, the 3rd row shows randomly horizontally and vertically flip the images, the last row shows images after adjust contrast, saturation and hue.
# * From this we can know, computer read images as set of numbers, which is different to human's eyes.

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:17:27.111113Z","iopub.execute_input":"2022-06-06T15:17:27.111614Z","iopub.status.idle":"2022-06-06T15:17:40.454608Z","shell.execute_reply.started":"2022-06-06T15:17:27.111577Z","shell.execute_reply":"2022-06-06T15:17:40.45331Z"}}


# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:04.209592Z","iopub.execute_input":"2022-06-06T15:18:04.210051Z","iopub.status.idle":"2022-06-06T15:18:06.323635Z","shell.execute_reply.started":"2022-06-06T15:18:04.21002Z","shell.execute_reply":"2022-06-06T15:18:06.322634Z"}}
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchsummary import summary

# %% [markdown]
# The normalization and other common data augmentation method is visualized on randomly sampled 5 images. All transformation applied after convert images to Numpy array, which is read by the Matploylib package. The transformation is performed by the Pytorch build in function, and transformed to PIL image, which is readable for Matploylib.

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:06.325834Z","iopub.execute_input":"2022-06-06T15:18:06.326747Z","iopub.status.idle":"2022-06-06T15:18:07.869283Z","shell.execute_reply.started":"2022-06-06T15:18:06.326706Z","shell.execute_reply":"2022-06-06T15:18:07.868123Z"}}
import PIL.Image as Image
pathset = []
fig, axs = plt.subplots(ncols=5, nrows=4, figsize=(15, 15))
for j in range(5):
    sample = birdSpecies.sample()
    species = sample['labels'].to_string(index=False)
    index = sample.index
    filepath = str(sample.filepaths.values)[2:-2]
    pathset.append(local_path + filepath)
    random_bird = mpimg.imread(local_path + filepath)
    plt.imshow(random_bird)
    axs[0, j].imshow(random_bird)
    axs[0, j].set_title(species)
    axs[0, j].axis('off')
    
    # Normalize RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        transforms.ToPILImage()
    ])
    pil_norm = transform(random_bird)
    axs[1, j].imshow(pil_norm)
    axs[1, j].axis('off')

    # Flip
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.ToPILImage()
    ])
    pil_flip = transform(random_bird)
    axs[2, j].imshow(pil_flip)
    axs[2, j].axis('off')

    # Color Jitter
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.ColorJitter(contrast=(0,5), saturation=(0,5), hue=(-0.1,0.1)),
      transforms.ToPILImage()
    ])
    pil_jitter = transform(random_bird)
    axs[3, j].imshow(pil_jitter)
    axs[3, j].axis('off')

# %% [markdown]
# The first row in the plot below shows the random select original image in the dataset; the second row shows the result after normalizing the numpy array of images; the third row shows random, horizontally and vertically, flipped images; the last row shows images after adjust contrast, saturation and hue.
# 
# The result is shown below. This shows computer reads images as set of numbers, which is different to human's eyes

# %% [markdown]
# # Benchmark

# %% [markdown]
# **Data Preprocessing**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:09.920923Z","iopub.execute_input":"2022-06-06T15:18:09.921481Z","iopub.status.idle":"2022-06-06T15:18:09.926721Z","shell.execute_reply.started":"2022-06-06T15:18:09.921449Z","shell.execute_reply":"2022-06-06T15:18:09.925336Z"}}
train_dir = 'D:/Python/datasets/bird/test'
test_dir = 'D:/Python/datasets/bird/test'
val_dir = 'D:/Python/datasets/bird/valid'

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:11.334588Z","iopub.execute_input":"2022-06-06T15:18:11.335008Z","iopub.status.idle":"2022-06-06T15:18:20.979337Z","shell.execute_reply.started":"2022-06-06T15:18:11.334978Z","shell.execute_reply":"2022-06-06T15:18:20.978108Z"}}
# All images has dimension in 224 X 224 X 3, in which the 3 represents RGB images. 
# The input dimension 224 x 224 is same as the model required. 
# Data is normalized while transform to pytorch datasets.

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root=train_dir, transform=transformations)
test_data = datasets.ImageFolder(root=test_dir, transform=transformations)
valid_data = datasets.ImageFolder(root=val_dir, transform=transformations)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:20.982203Z","iopub.execute_input":"2022-06-06T15:18:20.982664Z","iopub.status.idle":"2022-06-06T15:18:21.000206Z","shell.execute_reply.started":"2022-06-06T15:18:20.982625Z","shell.execute_reply":"2022-06-06T15:18:20.999119Z"}}
class_dict = pd.read_csv('D:/Python/datasets/bird/class_dict.csv')
classes = list(class_dict['class'])
print(len(classes))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:21.001657Z","iopub.execute_input":"2022-06-06T15:18:21.002679Z","iopub.status.idle":"2022-06-06T15:18:21.007489Z","shell.execute_reply.started":"2022-06-06T15:18:21.002633Z","shell.execute_reply":"2022-06-06T15:18:21.006355Z"}}
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 16

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:21.010448Z","iopub.execute_input":"2022-06-06T15:18:21.017593Z","iopub.status.idle":"2022-06-06T15:18:21.046927Z","shell.execute_reply.started":"2022-06-06T15:18:21.017543Z","shell.execute_reply":"2022-06-06T15:18:21.045663Z"}}
# obtain training indices that will be used for validation
num_train, num_valid = len(train_data), len(valid_data)
train_index, valid_index = list(range(num_train)), list(range(num_valid))
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)
# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         num_workers = num_workers)

# %% [markdown]
# **Train Test Setup**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:21.048581Z","iopub.execute_input":"2022-06-06T15:18:21.051206Z","iopub.status.idle":"2022-06-06T15:18:21.077083Z","shell.execute_reply.started":"2022-06-06T15:18:21.051161Z","shell.execute_reply":"2022-06-06T15:18:21.075686Z"}}
def train (n_epoch, base_model, train_loader, optimizer, criterion, valid_loader):
  for epoch in range(n_epoch):  # loop over the dataset multiple times

    print('Epoch: ', epoch+1)
    
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    train_loss = 0
    valid_loss = 0
    
    base_model.train()
    for images, labels in train_loader:
        
        images = images.to(device)
        
        #Reset Grads
        optimizer.zero_grad()
        
        #Forward ->
        outputs = base_model(images).to("cpu")
        
        #Calculate Loss & Backward, Update Weights (Step)
        loss = criterion(outputs, labels)
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    
    
    #Validation
    base_model.eval()
    for images, labels in valid_loader:
        
        images = images.to(device)
        
        #Forward ->
        preds = base_model(images).to("cpu")
        
        #Calculate Loss
        loss = criterion(preds, labels)
        valid_loss += loss.item() * images.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(preds, 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

        # calculate test accuracy for each object class
        for i in range(len(labels)):
            label = labels.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    print('\nValid Accuracy: %2d%%' % 
          (100. * np.sum(class_correct) / np.sum(class_total)))
    
    
    # print training/validation statistics 
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    
    #Print Epoch Statistics
    print("Train Loss = {}".format(round(train_loss, 4)))
    print("Valid Loss = {}".format(round(valid_loss, 4)))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:22:46.619793Z","iopub.execute_input":"2022-06-02T16:22:46.620981Z","iopub.status.idle":"2022-06-02T16:22:46.632631Z","shell.execute_reply.started":"2022-06-02T16:22:46.620932Z","shell.execute_reply":"2022-06-02T16:22:46.631677Z"}}
def test (classes, base_model, test_loader, criterion):
  # initialize lists to monitor test loss and accuracy
  test_loss = 0.0
  class_correct = list(0. for i in range(len(classes)))
  class_total = list(0. for i in range(len(classes)))
  base_model.eval() # prep model for evaluation

  for data, target in test_loader:

      data = data.to(device)

      # forward pass: compute predicted outputs by passing inputs to the model
      output = base_model(data).to("cpu")
      # calculate the loss
      loss = criterion(output, target)
      # update test loss 
      test_loss += loss.item()*data.size(0)

      # convert output probabilities to predicted class
      _, pred = torch.max(output, 1)

      # compare predictions to true label
      correct = np.squeeze(pred.eq(target.data.view_as(pred)))

      # calculate test accuracy for each object class
      for i in range(len(target)):
          label = target.data[i]
          class_correct[label] += correct[i].item()
          class_total[label] += 1
        
  # calculate and print avg test loss
  test_loss = test_loss/len(test_loader.sampler)
  print('Test Loss: {:.6f}\n'.format(test_loss))
  print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
      100. * np.sum(class_correct) / np.sum(class_total),
      np.sum(class_correct), np.sum(class_total)))

# %% [markdown]
# **Benchmark Model -- VGG 19**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:22:46.634924Z","iopub.execute_input":"2022-06-02T16:22:46.635426Z","iopub.status.idle":"2022-06-02T16:23:39.901548Z","shell.execute_reply.started":"2022-06-02T16:22:46.635401Z","shell.execute_reply":"2022-06-02T16:23:39.90072Z"}}
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.vgg19(pretrained=True)

# turn training false for all layers, other than fc layer
for param in base_model.parameters():
    param.requires_grad = False
    
base_model.classifier

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:23:39.902812Z","iopub.execute_input":"2022-06-02T16:23:39.903391Z","iopub.status.idle":"2022-06-02T16:23:40.82903Z","shell.execute_reply.started":"2022-06-02T16:23:39.903349Z","shell.execute_reply":"2022-06-02T16:23:40.828173Z"}}
base_model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 2048),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(2048, len(classes))
    )

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:23:40.830211Z","iopub.execute_input":"2022-06-02T16:23:40.83237Z","iopub.status.idle":"2022-06-02T16:23:43.927328Z","shell.execute_reply.started":"2022-06-02T16:23:40.832338Z","shell.execute_reply":"2022-06-02T16:23:43.926192Z"}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:23:43.931819Z","iopub.execute_input":"2022-06-02T16:23:43.932203Z","iopub.status.idle":"2022-06-02T16:23:43.941472Z","shell.execute_reply.started":"2022-06-02T16:23:43.932165Z","shell.execute_reply":"2022-06-02T16:23:43.94057Z"}}
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr = 0.0001)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:23:43.946695Z","iopub.execute_input":"2022-06-02T16:23:43.949517Z","iopub.status.idle":"2022-06-02T16:23:49.876749Z","shell.execute_reply.started":"2022-06-02T16:23:43.949478Z","shell.execute_reply":"2022-06-02T16:23:49.875843Z"}}
summary(base_model, (3,224,224))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T16:23:49.878187Z","iopub.execute_input":"2022-06-02T16:23:49.878538Z","iopub.status.idle":"2022-06-02T17:23:56.450855Z","shell.execute_reply.started":"2022-06-02T16:23:49.878509Z","shell.execute_reply":"2022-06-02T17:23:56.450005Z"}}
train(10, base_model, train_loader, optimizer, criterion, valid_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T17:23:56.452043Z","iopub.execute_input":"2022-06-02T17:23:56.453065Z","iopub.status.idle":"2022-06-02T17:24:20.976826Z","shell.execute_reply.started":"2022-06-02T17:23:56.453024Z","shell.execute_reply":"2022-06-02T17:24:20.975988Z"}}
test(classes, base_model, test_loader, criterion)







# %% [markdown]
# **Benchmark Model -- ResNet 18**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T17:24:20.977974Z","iopub.execute_input":"2022-06-02T17:24:20.980087Z","iopub.status.idle":"2022-06-02T17:24:27.307298Z","shell.execute_reply.started":"2022-06-02T17:24:20.980045Z","shell.execute_reply":"2022-06-02T17:24:27.30653Z"}}
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.resnet18(pretrained=True)

# turn training false for all layers, other than fc layer
for param in base_model.parameters():
    param.requires_grad = False
    
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, len(classes))
base_model = base_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr = 0.0005)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T17:24:27.308535Z","iopub.execute_input":"2022-06-02T17:24:27.308984Z","iopub.status.idle":"2022-06-02T17:24:27.475003Z","shell.execute_reply.started":"2022-06-02T17:24:27.308939Z","shell.execute_reply":"2022-06-02T17:24:27.474125Z"}}
summary(base_model, (3,224,224))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T17:24:27.476506Z","iopub.execute_input":"2022-06-02T17:24:27.477138Z","iopub.status.idle":"2022-06-02T17:57:07.625702Z","shell.execute_reply.started":"2022-06-02T17:24:27.477096Z","shell.execute_reply":"2022-06-02T17:57:07.624853Z"}}
train(10, base_model, train_loader, optimizer, criterion, valid_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T17:57:07.630147Z","iopub.execute_input":"2022-06-02T17:57:07.630786Z","iopub.status.idle":"2022-06-02T17:57:14.465059Z","shell.execute_reply.started":"2022-06-02T17:57:07.630754Z","shell.execute_reply":"2022-06-02T17:57:14.464183Z"}}
test(classes, base_model, test_loader, criterion)

# %% [markdown]
# **Which CNN architecture?**

# %% [markdown]
# The most popular models of CNN for image classification which input is dimension 3*224*224 are VGG and ResNet. Both two pre-trained models are applied as benchmarks.
# 
# Because of the extremely poor performance of VGG model, self-build CNN model will replace the VGG model and experimented with the ResNet-34 model in the following sections.

# %% [markdown]
# # Model Experiment

# %% [code] {"execution":{"iopub.status.busy":"2022-06-03T04:36:18.012284Z","iopub.execute_input":"2022-06-03T04:36:18.012718Z","iopub.status.idle":"2022-06-03T04:36:18.026544Z","shell.execute_reply.started":"2022-06-03T04:36:18.012687Z","shell.execute_reply":"2022-06-03T04:36:18.025438Z"}}
class bird_cnn(nn.Module):
  def __init__(self):
    super(bird_cnn, self).__init__()
    # input image channel = 3, output = 6, filter = 5x5
    self.conv1 = nn.Conv2d(3, 16, 5)
    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU() 
    self.pool1 = nn.MaxPool2d(2, 2)
    
    self.conv2 = nn.Conv2d(16, 32, 5)
    self.bn2 = nn.BatchNorm2d(32)
    self.relu = nn.ReLU() 
    self.pool2 = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(32*53*53, 2048)
    self.fc2 = nn.Linear(2048, 1024)
    # Output
    self.fc3 = nn.Linear(1024, len(classes))

  def forward(self, x):
    
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.pool2(x)
    
    x = x.view(-1, 32*53*53)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    
    return x

# %% [code] {"execution":{"iopub.status.busy":"2022-06-03T00:56:22.393254Z","iopub.execute_input":"2022-06-03T00:56:22.39364Z","iopub.status.idle":"2022-06-03T00:56:26.775223Z","shell.execute_reply.started":"2022-06-03T00:56:22.393591Z","shell.execute_reply":"2022-06-03T00:56:26.774391Z"}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bird_cnn().to(device)

### Training Details

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# The self-build CNN model is based on the idea of VGG, but a simplified version. The architecture is shown below, which generated by the Pytorch summary.

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T17:57:16.159234Z","iopub.execute_input":"2022-06-02T17:57:16.159586Z","iopub.status.idle":"2022-06-02T17:57:16.171615Z","shell.execute_reply.started":"2022-06-02T17:57:16.159551Z","shell.execute_reply":"2022-06-02T17:57:16.170736Z"}}
summary(model, (3,224,224))

# %% [markdown]
#  In VGG-19, each CNN block has two convolution layers, in this model, there is one in each block. To avoid overfitting, this model introduced the “BarchNorm” layer before the “ReLU” activation layer, in stead of the dropout used in the fully connected layer in VGG. The input shape is [3, 224, 224], then after the first convolution layer with output channel set to 6 and 5x5 filter, the output shape is [6, 220, 220].

# %% [markdown]
# Moving on to the ResNet, the reason that ResNet has outstanding accuracy in the benchmark is that this network considered the residual. The residual blocks in the architecture can capture the residual as F(x), which works by add the input to the output and skipped the layer between. Compared with regular CNN architecture, where layers stacked and connected, skip-connection improve the possible layer from 16 or 19 in VGG to 34, 50, 101 and more in the ResNet by saving computational load.

# %% [markdown]
# In stead of using build-in architecture in Pytorch, customized ResNet is used in the project, which brings the flexibility for future fine tuning. The residual blocks in ResNet-18 and ResNet-34 is Basic Block, and in deeper learning is BottleNeck Blocks. In this project, Basic Block, reference from **“jarvislabs.ai”** is applied to customize the ResNet-34, as the selected network architecture.

# %% [markdown]
# https://jarvislabs.ai/blogs/resnet

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:26.416772Z","iopub.execute_input":"2022-06-06T15:18:26.417212Z","iopub.status.idle":"2022-06-06T15:18:26.424894Z","shell.execute_reply.started":"2022-06-06T15:18:26.417183Z","shell.execute_reply":"2022-06-06T15:18:26.423546Z"}}
conv_block = nn.Sequential(nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False),  # 112x112
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # 112/2 = 56

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:27.469895Z","iopub.execute_input":"2022-06-06T15:18:27.470571Z","iopub.status.idle":"2022-06-06T15:18:27.483765Z","shell.execute_reply.started":"2022-06-06T15:18:27.470538Z","shell.execute_reply":"2022-06-06T15:18:27.482191Z"}}
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        ## in example, input channel is 64, output channel is 128
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                    padding=1, bias=False) #kernel size same as output from basic layers
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                    padding=1, bias=False) #kernel size same as output from basic layers
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:29.619468Z","iopub.execute_input":"2022-06-06T15:18:29.620719Z","iopub.status.idle":"2022-06-06T15:18:29.636883Z","shell.execute_reply.started":"2022-06-06T15:18:29.620687Z","shell.execute_reply":"2022-06-06T15:18:29.635643Z"}}
def _make_layer(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
        )
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:31.248578Z","iopub.execute_input":"2022-06-06T15:18:31.248924Z","iopub.status.idle":"2022-06-06T15:18:31.268104Z","shell.execute_reply.started":"2022-06-06T15:18:31.248898Z","shell.execute_reply":"2022-06-06T15:18:31.266835Z"}}
class ResNet(nn.Module):

    def __init__ (self, block, layers, num_classes=len(classes)):
        super().__init__()

        self.inplanes = 64

        # beginning layer without residual
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 sequentials of residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])  # how many conv layer in corresponding layer
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # end layer to linear
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)  # 224x224
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.maxpool(x) # 112x112

        x = self.layer1(x) # 56x56
        x = self.layer2(x) # 28x28
        x = self.layer3(x) # 14x14
        x = self.layer4(x) # 7x7

        x = self.avgpool(x) # 1x1
        x = torch.flatten(x, 1) # remove 1x1 grid and make vector of tensor shape
        x = self.fc(x)

        return x

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:34.315295Z","iopub.execute_input":"2022-06-06T15:18:34.31667Z","iopub.status.idle":"2022-06-06T15:18:34.325193Z","shell.execute_reply.started":"2022-06-06T15:18:34.316625Z","shell.execute_reply":"2022-06-06T15:18:34.323741Z"}}
def resnet34_build():
    layers = [3,4,6,3]
    model = ResNet(BasicBlock, layers)
    return model

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:35.320719Z","iopub.execute_input":"2022-06-06T15:18:35.321265Z","iopub.status.idle":"2022-06-06T15:18:38.969555Z","shell.execute_reply.started":"2022-06-06T15:18:35.321234Z","shell.execute_reply":"2022-06-06T15:18:38.968349Z"}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_res = resnet34_build().to(device)

### Training Details

optimizer = torch.optim.Adam(model_res.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()


# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T15:47:05.816134Z","iopub.execute_input":"2022-06-04T15:47:05.816596Z","iopub.status.idle":"2022-06-04T15:47:12.034214Z","shell.execute_reply.started":"2022-06-04T15:47:05.81655Z","shell.execute_reply":"2022-06-04T15:47:12.033373Z"}}
summary(model_res, (3,224,224))

# %% [markdown]
# **Train Test Setup**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:38.974474Z","iopub.execute_input":"2022-06-06T15:18:38.974854Z","iopub.status.idle":"2022-06-06T15:18:38.982476Z","shell.execute_reply.started":"2022-06-06T15:18:38.974826Z","shell.execute_reply":"2022-06-06T15:18:38.981118Z"}}
import time
import seaborn as sns

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:39.163595Z","iopub.execute_input":"2022-06-06T15:18:39.163894Z","iopub.status.idle":"2022-06-06T15:18:39.18335Z","shell.execute_reply.started":"2022-06-06T15:18:39.163867Z","shell.execute_reply":"2022-06-06T15:18:39.182124Z"}}
def train(epochs, num_class, model, train_loader, valid_loader, optimizer=optimizer, criterion=criterion):
    result = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        since = time.time()

        print('Epoch: ', epoch+1)
        
        class_correct = list(0. for i in range(num_class))
        class_total = list(0. for i in range(num_class))
        train_loss = 0
        valid_loss = 0
        
        model.train()
        for images, labels in train_loader:
            
            images = images.to(device)
            
            #Reset Grads
            optimizer.zero_grad()
            
            #Forward ->
            outputs = model(images).to("cpu")
            
            #Calculate Loss & Backward, Update Weights (Step)
            loss = criterion(outputs, labels)
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        
        #Validation
        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                
                images = images.to(device)
                
                #Forward ->
                preds = model(images).to("cpu")
                
                #Calculate Loss
                loss = criterion(preds, labels)
                valid_loss += loss.item() * images.size(0)

                # convert output probabilities to predicted class
                _, pred = torch.max(preds, 1)

                # compare predictions to true label
                correct = np.squeeze(pred.eq(labels.data.view_as(pred)))

                # calculate test accuracy for each object class
                for i in range(len(labels)):
                    label = labels.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
        valid_acc = 100. * np.sum(class_correct) / np.sum(class_total)
        print('\nValid Accuracy: %2d%%' % 
              (valid_acc))

        
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        
        #Print Epoch Statistics
        print("Train Loss = {}".format(round(train_loss, 4)))
        print("Valid Loss = {}".format(round(valid_loss, 4)))

        result.append([train_loss, valid_loss, valid_acc])

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    
    return result

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:41.631973Z","iopub.execute_input":"2022-06-06T15:18:41.632573Z","iopub.status.idle":"2022-06-06T15:18:41.644284Z","shell.execute_reply.started":"2022-06-06T15:18:41.632541Z","shell.execute_reply":"2022-06-06T15:18:41.643125Z"}}
def test (classes, model, test_loader, criterion=criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    model.eval() # prep model for evaluation

    for data, target in test_loader:

        data = data.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to("cpu")
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
          
    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    test_acc = 100. * np.sum(class_correct) / np.sum(class_total)

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        test_acc,
        np.sum(class_correct), np.sum(class_total)))

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:44.037623Z","iopub.execute_input":"2022-06-06T15:18:44.037988Z","iopub.status.idle":"2022-06-06T15:18:44.046228Z","shell.execute_reply.started":"2022-06-06T15:18:44.037959Z","shell.execute_reply":"2022-06-06T15:18:44.044469Z"}}
def plot_metrics(result):
    result['epochs'] = np.array([1,2,3,4,5,6,7,8,9,10])
    long = (result.drop(['valid_acc'], axis=1)).melt('epochs', var_name='metrics', value_name='values')
    sns.lineplot(x='epochs', y="values", hue='metrics', data=long)
    ax2 = plt.twinx()
    sns.lineplot(data=result, x='epochs', y='valid_acc', ax=ax2, color='red', label='valid_acc')
    plt.show()

# %% [markdown]
# **CNN Model**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-03T00:57:09.390586Z","iopub.execute_input":"2022-06-03T00:57:09.390972Z","iopub.status.idle":"2022-06-03T01:36:22.09734Z","shell.execute_reply.started":"2022-06-03T00:57:09.390939Z","shell.execute_reply":"2022-06-03T01:36:22.0965Z"}}
result = train(10, len(classes), model, train_loader, valid_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-03T01:36:22.098911Z","iopub.execute_input":"2022-06-03T01:36:22.099637Z","iopub.status.idle":"2022-06-03T01:36:38.79214Z","shell.execute_reply.started":"2022-06-03T01:36:22.099596Z","shell.execute_reply":"2022-06-03T01:36:38.791287Z"}}
test(classes, model, test_loader, criterion)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-03T01:36:38.79334Z","iopub.execute_input":"2022-06-03T01:36:38.794132Z","iopub.status.idle":"2022-06-03T01:36:38.802919Z","shell.execute_reply.started":"2022-06-03T01:36:38.794092Z","shell.execute_reply":"2022-06-03T01:36:38.802136Z"}}
result_df = pd.DataFrame(result, columns=['train_loss','valid_loss','valid_acc'])
result_df.to_csv("/kaggle/working/result_df.csv")

# %% [code] {"execution":{"iopub.status.busy":"2022-06-03T01:36:38.806188Z","iopub.execute_input":"2022-06-03T01:36:38.806954Z","iopub.status.idle":"2022-06-03T01:36:39.117135Z","shell.execute_reply.started":"2022-06-03T01:36:38.806913Z","shell.execute_reply":"2022-06-03T01:36:39.11644Z"}}
plot_metrics(result_df)

# %% [markdown]
# **ResNet34**

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T18:28:57.490702Z","iopub.execute_input":"2022-06-02T18:28:57.491262Z","iopub.status.idle":"2022-06-02T19:16:31.395954Z","shell.execute_reply.started":"2022-06-02T18:28:57.491225Z","shell.execute_reply":"2022-06-02T19:16:31.39508Z"}}
result_res = train(10, len(classes), model_res, train_loader, valid_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T19:16:31.397417Z","iopub.execute_input":"2022-06-02T19:16:31.399081Z","iopub.status.idle":"2022-06-02T19:16:39.030283Z","shell.execute_reply.started":"2022-06-02T19:16:31.399039Z","shell.execute_reply":"2022-06-02T19:16:39.028853Z"}}
test(classes, model_res, test_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T19:16:39.031925Z","iopub.execute_input":"2022-06-02T19:16:39.032583Z","iopub.status.idle":"2022-06-02T19:16:39.040051Z","shell.execute_reply.started":"2022-06-02T19:16:39.032537Z","shell.execute_reply":"2022-06-02T19:16:39.039214Z"}}
result_res_df = pd.DataFrame(result_res, columns=['train_loss','valid_loss','valid_acc'])
result_res_df.to_csv("./result_res_df.csv")

# %% [code] {"execution":{"iopub.status.busy":"2022-06-02T19:16:39.041727Z","iopub.execute_input":"2022-06-02T19:16:39.042292Z","iopub.status.idle":"2022-06-02T19:16:39.42365Z","shell.execute_reply.started":"2022-06-02T19:16:39.042251Z","shell.execute_reply":"2022-06-02T19:16:39.422874Z"}}
plot_metrics(result_res_df)

# %% [markdown]
# The results of the evaluation metrics of both models are shown above. Although the CNN model is very simple, with only two convolutional blocks, the valid accuracy reaches 55%, and the test accuracy reaches 57% for 400 classes.
# 
# Comparing the performance of CNN and ResNet-34, the time spent doesn’t have a significant difference, both models cost around 6 hours in the first epoch training. But the training loss decreases faster in the ResNet-34 model after the second epoch and the test loss shows a huge drop after the first epoch. The plot of ResNet-34 shows a clearer trend that loss and accuracy are negatively correlated, when loss decreases significantly after the first epoch training, the valid accuracy jump from 18% to 50%. However, in the plot of the CNN model, the valid accuracy shows unstable increasing.
# 
# The test accuracy of the ResNet-34 model reaches 91%. However, both model shows an increase in validation loss, which is the symbol of overfitting. Therefore, the following step is introducing the data augmentation to avoid overfitting, and wish can have further improvement in the metrics of the classification.

# %% [markdown]
# **Data Augmentation**

# %% [markdown]
# In the previous section, ResNet-34 has a better performance compared with the CNN model, so the data augmentation is simply applied to the ResNet-34 model.
# 
# There are two transformations experimented within this section, horizontal flipping and color jitter, and both transformations are applied one by one.

# %% [markdown]
# - Horizontal Flipping

# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T15:47:20.206019Z","iopub.execute_input":"2022-06-04T15:47:20.206382Z","iopub.status.idle":"2022-06-04T15:47:21.245985Z","shell.execute_reply.started":"2022-06-04T15:47:20.206346Z","shell.execute_reply":"2022-06-04T15:47:21.245164Z"}}
# All image is 224 X 224 X 3, no need resiez eand centercrop

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
valid_data = datasets.ImageFolder(root=val_dir, transform=test_transform)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T15:47:21.86142Z","iopub.execute_input":"2022-06-04T15:47:21.862208Z","iopub.status.idle":"2022-06-04T15:47:21.872942Z","shell.execute_reply.started":"2022-06-04T15:47:21.862171Z","shell.execute_reply":"2022-06-04T15:47:21.872117Z"}}
# obtain training indices that will be used for validation
num_train, num_valid = len(train_data), len(valid_data)
train_index, valid_index = list(range(num_train)), list(range(num_valid))
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)
# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         num_workers = num_workers)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T15:47:24.328444Z","iopub.execute_input":"2022-06-04T15:47:24.328831Z","iopub.status.idle":"2022-06-04T16:40:35.41899Z","shell.execute_reply.started":"2022-06-04T15:47:24.3288Z","shell.execute_reply":"2022-06-04T16:40:35.418121Z"}}
result_res_aug = train(10, len(classes), model_res, train_loader, valid_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T16:40:35.42117Z","iopub.execute_input":"2022-06-04T16:40:35.42185Z","iopub.status.idle":"2022-06-04T16:40:52.144779Z","shell.execute_reply.started":"2022-06-04T16:40:35.42181Z","shell.execute_reply":"2022-06-04T16:40:52.143957Z"}}
test(classes, model_res, test_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T16:40:52.146013Z","iopub.execute_input":"2022-06-04T16:40:52.147759Z","iopub.status.idle":"2022-06-04T16:40:52.153606Z","shell.execute_reply.started":"2022-06-04T16:40:52.147721Z","shell.execute_reply":"2022-06-04T16:40:52.152776Z"}}
result_res_aug_df = pd.DataFrame(result_res_aug, columns=['train_loss','valid_loss','valid_acc'])

# %% [code] {"execution":{"iopub.status.busy":"2022-06-04T16:40:52.155502Z","iopub.execute_input":"2022-06-04T16:40:52.155929Z","iopub.status.idle":"2022-06-04T16:40:52.473444Z","shell.execute_reply.started":"2022-06-04T16:40:52.155891Z","shell.execute_reply":"2022-06-04T16:40:52.472611Z"}}
plot_metrics(result_res_aug_df)

# %% [markdown]
# - Color Jitter

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:53.630617Z","iopub.execute_input":"2022-06-06T15:18:53.631123Z","iopub.status.idle":"2022-06-06T15:18:54.915132Z","shell.execute_reply.started":"2022-06-06T15:18:53.631078Z","shell.execute_reply":"2022-06-06T15:18:54.914083Z"}}
# All image is 224 X 224 X 3, no need resiez eand centercrop

train_transform = transforms.Compose([
    transforms.ColorJitter(contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
valid_data = datasets.ImageFolder(root=val_dir, transform=test_transform)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:55.137932Z","iopub.execute_input":"2022-06-06T15:18:55.13874Z","iopub.status.idle":"2022-06-06T15:18:55.154808Z","shell.execute_reply.started":"2022-06-06T15:18:55.138708Z","shell.execute_reply":"2022-06-06T15:18:55.153472Z"}}
# obtain training indices that will be used for validation
num_train, num_valid = len(train_data), len(valid_data)
train_index, valid_index = list(range(num_train)), list(range(num_valid))
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)
# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         num_workers = num_workers)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T15:18:56.733629Z","iopub.execute_input":"2022-06-06T15:18:56.734273Z","iopub.status.idle":"2022-06-06T17:07:39.512295Z","shell.execute_reply.started":"2022-06-06T15:18:56.734243Z","shell.execute_reply":"2022-06-06T17:07:39.511118Z"}}
result_res_aug2 = train(10, len(classes), model_res, train_loader, valid_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T17:07:39.514585Z","iopub.execute_input":"2022-06-06T17:07:39.515287Z","iopub.status.idle":"2022-06-06T17:07:56.400478Z","shell.execute_reply.started":"2022-06-06T17:07:39.515239Z","shell.execute_reply":"2022-06-06T17:07:56.399395Z"}}
test(classes, model_res, test_loader)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T17:07:56.402211Z","iopub.execute_input":"2022-06-06T17:07:56.40281Z","iopub.status.idle":"2022-06-06T17:07:56.415258Z","shell.execute_reply.started":"2022-06-06T17:07:56.402753Z","shell.execute_reply":"2022-06-06T17:07:56.413839Z"}}
result_res_aug2_df = pd.DataFrame(result_res_aug2, columns=['train_loss','valid_loss','valid_acc'])
result_res_aug2_df.to_csv("/kaggle/working/result_res_aug2_df.csv")

# %% [code] {"execution":{"iopub.status.busy":"2022-06-06T17:07:56.417939Z","iopub.execute_input":"2022-06-06T17:07:56.418452Z","iopub.status.idle":"2022-06-06T17:07:56.930927Z","shell.execute_reply.started":"2022-06-06T17:07:56.418414Z","shell.execute_reply":"2022-06-06T17:07:56.929984Z"}}
plot_metrics(result_res_aug2_df)

# %% [markdown]
# Comparing the horizontal flipping and the color jitter, the color jitter transformed data has lower accuracy in both valid and test procedures, converge slower, and is more time-consuming. Therefore, horizontal flipping is the appropriate transformation for this dataset, with the test accuracy improved by two percent.

# %% [markdown]
# The reason that color jitter performance is worse might be that the color is more important when distinguishing bird species. For example, some species will have a special color in a specific part of the body, such as heads or tails. Therefore, color jitter transformation makes the dataset harder to be distinguished and costs longer time in the first training epoch. Although both models perform well on avoiding overfitting, the data augmentation is perhaps a trade-off between the differentiability of the dataset and the overfitting of the training.

# %% [markdown]
# # Conclusion

# %% [markdown]
# Efficient and reliable monitoring of bird species can reveal whether the ecology of the habits is well maintained or threatened. Although machines have the advantage of processing large information, it cannot extract message from images like human eyes.
# 
# The convolutional neural network built in this project can help train the machine to understand the information beyond the pixels. From the experiment among different neural networks and different augment methods, the best model is the ResNet-34 model combines with horizontal flipping data augmentation. Using this model, 93% of images can be correctly classified, which is meant to facilitate ecologists overcome the obstacle of classifying large volume of images. This can benefit researchers working on time-sensitive topics, such as ecologists working on the protection of endangered bird species.
# 
# However, the performance of the model reliability of the model needs further improvement, and the generalization is necessary before applying it to the industry. Some possible improvements will be introduced in the next section.

# %% [markdown]
# **Future Improvement**

# %% [markdown]
# - Hyperparameter tuning

# %% [markdown]
# Batch size: The batch size affects the result of the loss function. In addition to 16, the larger batch size such as 32, 64 or 128 can be experimented.
# 
# Optimizer: Although Adam is one of the most commonly used optimizers, which converge fast, the drawback is the lower final performance. Therefore, SGD is an alternative option and can be experimented with different momentum to control the convergence speed.
# 
# Epoch: More epochs can be performed to further improve the accuracy.

# %% [markdown]
# Although this project has tested basic CNN and ResNet-34 models, there are still other models in the ResNet family worth the experiment, such as the deeper ResNet model with bottleneck residual blocks and ResNeXt with hyperparameter cardinality.

# %% [markdown]
# - YOLO Image Caption

# %% [markdown]
# The explanatory data analysis session shows the characteristics of this dataset, with all images cropped and birds centered. In other words, on which kind of dataset this model can reach the performance similar to the result is shown in the model implementation section. However, this is a strict requirement for the real-world dataset.
# 
# The business goal this project wants to achieve is offering a tool that can automatically, accurately, and inexpensively monitor the bird species. Therefore, to fully realize the inexpensively and automatically, instead of prepared images, it is ideal that ecologists or researchers can use images from “camera traps” directly for the classification of the bird species. This further step needs pre-processing by YOLO image captioning.
# 
# This image captioning model can capture the figure user desired with the noise in the background, which makes the image captured from conserve ecosystems applicable to this model. Moreover, with the ability to automatically recognize the bird, this model can be used for online active training with any user-uploaded images to help improve the accuracy further.

# %% [markdown]
# # Reference

# %% [markdown]
# Rachit Tayal. (2020). Deep Learning for Computer Vision.
# https://towardsdatascience.com/deep-learning-for-computer-vision-c4e5f191c522
# Poonamvligade. (2020). Building ResNet34 in PyTorch.
# https://github.com/jarvislabsai/blog/blob/master/build_resnet34_pytorch/Building%20Resnet% 20in%20PyTorch.ipynb
# Purva91. (2020). Top 4 Pre-Trained Models for Image Classification with Python Code.
# https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-
# classification-with-python-code
# Jason Brownlee. (2020). How to Control the Stability of Training Neural Networks With the Batch Size
# https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-
# neural-networks-with-gradient-descent-batch-size/
# Mohammad Sadegh Norouzzadeh, Anh Nguyen, Margaret Kosmala. (2018) Automatically identifying,
# counting, and describing wild animals in camera-trap images with deep learning
# https://www.pnas.org/doi/abs/10.1073/pnas.1719367115

# %% [markdown]
# **This is the final project of NEU IE7275, and thanks professor Arasu providing solid help through the semester.**

# %% [markdown]
# **The code is here** https://github.com/JuneHou/BirdSpeciesCNN

# %% [markdown]
# I welcome any comments you might like to make. :)