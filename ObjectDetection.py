# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:42:16 2022

@author: nightrain
"""

"""
failed 

"""
#%%
import os, time, random
import numpy as np
import pandas as pd
import cv2, torch
from tqdm.auto import tqdm
import shutil as sh

from IPython.display import Image, clear_output
import matplotlib.pyplot as plt
#%%
path="D:\Python\datasets\ObjectDetection\\"
#%%
img_h, img_w, num_channels = (380, 676, 3)
df = pd.read_csv(path+"train_solution_bounding_boxes (1).csv")
df.rename(columns={'image':'image_id'}, inplace=True)
df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])
df['x_center'] = (df['xmin'] + df['xmax'])/2
df['y_center'] = (df['ymin'] + df['ymax'])/2
df['w'] = df['xmax'] - df['xmin']
df['h'] = df['ymax'] - df['ymin']
df['classes'] = 0
df['x_center'] = df['x_center']/img_w
df['w'] = df['w']/img_w
df['y_center'] = df['y_center']/img_h
df['h'] = df['h']/img_h
#%%
path2="D:\Python\datasets\ObjectDetection\\training_images\\"
index = list(set(df.image_id))
image = random.choice(index)
print("Image ID: %s"%(image))
img = cv2.imread(path2+image+'.jpg')
#%%
image = random.choice(index)
Image(filename=path2+image+'.jpg',width=600)
#%%





