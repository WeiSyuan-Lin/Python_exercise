# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:56:45 2022

@author: nightrain
"""

"""
FastAI is built on Pytorch, NumPy, PIL, pandas, and a few other libraries. 
To achieve its goals, it does not aim to hide the lower levels of its foundation. 
Using this machine learning library, 
we can directly interact with the underlying PyTorch primitive models.

"""
#%%
import fastbook
fastbook.setup_book()
#%%
from fastai.vision.all import *
path = untar_data(URLs.PETS)
#%%
def is_cat(x):
  return x[0].isupper()
#%%
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct = 0.2,
    seed = 42,
    label_func = is_cat,
    item_tfms = Resize(224)
)
#%%
learn =cnn_learner(dls,
                   resnet34,
                   metrics = error_rate)

learn.fine_tune(1)
import ipywidgets as widgets

uploader = widgets.FileUpload()
uploader
def pred():
  img = PILImage.create(uploader.data[0])
  img.show()

  #Make Prediction
  is_cat,_,probs = learn.predict(img)

  print(f"Image is of a Cat: {is_cat}.")
  print(f"Probability image is a cat: {probs[1].item():.6f}")
  pred()
#%%











