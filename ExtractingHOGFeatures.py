# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:24:46 2022

@author: nightrain
"""

from skimage import  data, color, feature
image = color.rgb2gray(data.chelsea())
hogVec, hogVis = feature.hog(image, visualize=True)
import matplotlib.pyplot as plt
#%%
fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[],
                                       yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')
ax[1].imshow(hogVis)
ax[1].set_title("extarcting features from image")
plt.show()