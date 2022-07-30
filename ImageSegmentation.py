# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:22:07 2022

@author: nightrain
"""

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
#%%
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
#%%
