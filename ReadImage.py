# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 18:39:33 2022

@author: nightrain
"""
import argparse
import os
from PIL import Image as Img
import numpy as np
import matplotlib.pyplot as plt
#%%
#pathFolder="C:\\Users\\nightrain\\Desktop\\TestPics"
#%%
def LoadImgs(pathFolder):
    for filename in os.listdir(pathFolder):
        print(pathFolder+"\\"+filename)
        img=Img.open(pathFolder+"\\"+filename)
        #img.show()
        img=np.asarray(img)
        print(type(img))
        plt.figure()
        plt.imshow(img)
    plt.show()
#%%
if __name__== "__main__":
    parser = argparse.ArgumentParser(description='load images')
    parser.add_argument('pathFolder', type=str,help='image folder path')
  
    args = parser.parse_args()
    LoadImgs(args.pathFolder)