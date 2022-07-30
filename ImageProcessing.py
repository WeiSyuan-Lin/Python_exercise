# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 18:39:33 2022

@author: nightrain
"""
#import argparse
import os
from PIL import Image as Img
import numpy as np
import matplotlib.pyplot as plt
#%%
pathFolder="C:\\Users\\nightrain\\Desktop\\TestPics"
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
#LoadImgs(pathFolder)    
path="D:\Python\datasets\TestPics\Lenna.jpg"
#%%
img=Img.open(path).convert('L')
img=np.asarray(img,dtype=float)
#img=np.array(img)
print(type(img))
plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
#%%
def BiLevel_thresholding(img,c=113):
    """
    圖片呈現兩極化
    """
    height,width=img.shape
    
    temp=img.copy().reshape(height*width,-1)
    
    Ind=np.where(temp>c)[0]
    temp=np.zeros_like(temp)
    temp[Ind]=225*np.ones_like(Ind).reshape(len(Ind),-1)
    temp=temp.reshape(height,width)
    return temp
def ImageNegative(img):
    """
    圖片呈現負片效果
    """
    temp=img*-1+225
    return temp
#%%
temp=BiLevel_thresholding(img,c=80)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
temp=ImageNegative(img)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
# height,width=img.shape
# n=height*width

# pixV=np.arange(0,256)
# counts=[]
# for i in range(256):
#     counts.append(np.count_nonzero(img==i))
# counts=np.array(counts,dtype=int)
# plt.plot(pixV,counts/n)

#%%
def UniformHistogram(img):
    """
    改變圖形grayscale 分布 成接近均勻分布分布
    """
    counts=[]
    height,width=img.shape
    n=height*width
    temp=img.copy().reshape(height*width,-1)
    temp=temp.astype(float)
    for i in range(256):
        counts.append(np.count_nonzero(temp==i))
    counts=np.array(counts,dtype=int)
    
    for k in range(256):
        
        Ind=np.where(temp==k)[0]
        if len(Ind)>0:
            temp[Ind]=225*(counts[:k].sum()/n)*np.ones_like(Ind).reshape(len(Ind),-1)
        
    temp=temp.reshape(height,width)
    return temp
#%%
temp=UniformHistogram(img)
plt.imshow(temp, cmap=plt.get_cmap('gray'))

#%%
"""
filter operator

"""
from scipy.signal import convolve2d, medfilt
#%%
kernel = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])/16
kernel = np.array([[1., 2.,4.,2, 1.], [2., 4.,10.,4., 2.], [4.,10.,16.,10.,4.], [2., 4.,10.,4., 2.]
                  ,[1., 2.,4.,2, 1.]])/108
#%%
h,w=img.shape
noise=np.random.rand(h,w)
imgN=img+noise*225
plt.imshow(imgN, cmap=plt.get_cmap('gray'))
#%%
# gaussian filter
imgN=convolve2d(imgN,kernel,boundary='symm', mode='same')
plt.imshow(imgN, cmap=plt.get_cmap('gray'))
#%%
# median filter
imgN=medfilt(imgN,3)
plt.imshow(imgN, cmap=plt.get_cmap('gray'))

#%%
# basic highpass spatial filter
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # 補零
    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image
        
        
    # 確定有效範圍 和 正確的convoluiton間距
    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output
#%%
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/9
temp=convolve2D(img, kernel, padding=2)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
def BasicHighpass(k):
    kernel=-1*np.ones([3,3])
    kernel[1,1]=9*k-1
    kernel=kernel/(9*k)
    return kernel

def HighBoost(k):
    kernel=-1*np.ones([3,3])
    kernel[1,1]=9*k-1
    kernel=kernel/(9)
    return kernel

def HighFreqEmphasis(k):
    t=1-k
    s=8*k+1
    kernel=t*np.ones([3,3])
    kernel[1,1]=s
    kernel=kernel/(9)
    return kernel
#%%
k=1
kernel=BasicHighpass(k)
temp=convolve2D(img, kernel, padding=2)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
k=1.5
kernel=HighBoost(k)
temp=convolve2D(img, kernel, padding=0)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
k=10
kernel=HighFreqEmphasis(k)
temp=convolve2D(img, kernel, padding=0)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
def GradFilter(direc):
    kernel_x = np.array([[0., 0., 0.], [-1., 1., 0.], [0., 0., 0.]])
    kernel_y = np.array([[0., 0., 0.], [0., -1., 0.], [0., 1., 0.]])
    if direc=='x':
        kernel=kernel_x
    elif direc=='y':
        kernel=kernel_y
    else:
        return print('wrong')
        
    return kernel
#%%
kernel=GradFilter(direc='y')  
temp=convolve2D(img, kernel, padding=0)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
#Laplacian operator (sharpness highpass filter)
kernel_L = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]) 
temp=convolve2D(img, kernel_L, padding=0)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
# the norm of gradient
kernelx=GradFilter(direc='x')  
tempx=convolve2D(img, kernelx, padding=0)
kernely=GradFilter(direc='y')  
tempy=convolve2D(img, kernely, padding=0)
temp=np.sqrt(tempx**2+tempy**2)
plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%
# use Foutier transform to image processing

from numpy.fft import fftshift,ifftshift,fft2,ifft2
#%%
def calculate_2dft(img):
    ft = ifftshift(img)
    ft = fft2(ft)
    return fftshift(ft)
def calculate_2dift(img):
    ift = ifftshift(img)
    ift = ifft2(ift)
    ift = fftshift(ift)
    return ift.real
#%%
# highpass (lowpass) filter

# Calculate Fourier transform
Fimg=calculate_2dft(img)

h,w=Fimg.shape
r0=20
temp=np.zeros_like(Fimg)
for u in range(-int(w/2),int(w/2)):
    for v in range(-int(h/2),int(h/2)):
        r=np.sqrt(u**2+v**2)
        if r>r0:
            print(r)
            temp[u+int(w/2)][v+int(h/2)]=Fimg[u+int(w/2)][v+int(h/2)]

# Calculate inverse Fourier transform           
temp=calculate_2dift(temp)

plt.imshow(temp, cmap=plt.get_cmap('gray'))
#%%













