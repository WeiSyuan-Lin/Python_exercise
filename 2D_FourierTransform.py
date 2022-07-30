# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 00:43:09 2022

@author: nightrain
"""
"""
reference

https://thepythoncodingbook.com/2021/08/30/
 2d-fourier-transform-in-python-and-fourier-synthesis-of-images/
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

# construct sine wave

x = np.arange(-500, 501, 1)

X, Y = np.meshgrid(x, x)

wavelength = 200
angle = 0
grating = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength
)

plt.set_cmap("gray")

#plt.subplot(121)
plt.imshow(grating)
#%%
# Calculate Fourier transform of grating
ft = np.fft.ifftshift(grating) # 移動time space的點的順序，不加也對，要注意順序
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft) # 將 0 frequency 放中間，-N/2,...,0,...N/2

#plt.subplot(122)
plt.imshow(abs(ft))
plt.xlim([480, 520])
plt.ylim([520, 480])  # Note, order is reversed for y
plt.show()
#%%
# Calculate inverse Fourier transform
# 可以 Fourier transform 抵銷
ft = np.fft.ifftshift(ft)
ft = np.fft.ifft2(ft)
p = np.fft.fftshift(ft).real # 加這一行呈現效果較好, zero-frequency在圖形中間
plt.imshow(p)
#%%
max(abs(p-grating).flatten())
#%%
