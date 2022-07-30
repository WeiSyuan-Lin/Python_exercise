# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 21:35:51 2022

@author: nightrain
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly
#%%
path="D:\Python\datasets\ObjectImage\\cars.jpg"
#%%
image = cv2.imread(path)
box, label, count = cv.detect_common_objects(image)
output = draw_bbox(image, box, label, count)
plt.imshow(output)
plt.show()
print("Number of cars in this image are " +str(label.count('car')))
