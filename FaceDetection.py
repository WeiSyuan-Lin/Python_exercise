# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:29:06 2022

@author: nightrain
"""

import cv2
#%%
path='D:\\Python\\datasets\\FaceDetection\\face_detector.xml'
#%%
face_cascade = cv2.CascadeClassifier(path)
#%%
path_img='D:\\Python\\datasets\\FaceDetection\\image.png'
#%%
img = cv2.imread(path_img)
#%%
faces = face_cascade.detectMultiScale(img, 1.1, 4)
#%%
for (x, y, w, h) in faces: 
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite("face_detected.png", img) 
print('Successfully saved')
#%%