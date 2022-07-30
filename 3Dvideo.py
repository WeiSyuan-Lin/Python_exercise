# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 22:18:43 2022

@author: nightrain
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# create data points
x = np.linspace(-10, 10, 100)
y = np.linspace(-15, 15, 100)
# create grid
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection = '3d')
# hide the axis
ax._axis3don = False
# 3d contour plot
ax.contour3D(X, Y, Z, 100, cmap = 'viridis')
# make panel color white
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make grid color white
ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
# adjust point of view
ax.view_init(60, 90)
#%%
# adjust point of view
ax.view_init(60, 90)
#%%
for angle in range(0,360,2):
    plt.figure(figsize=(16, 9))
    ax = plt.axes(projection = '3d')
    ax._axis3don = False
    # 3d contour plot
    ax.contour3D(X, Y, Z, 200, cmap = 'viridis')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
    
    # adjust view from 0, 2, 4, 6, ..., 360
    ax.view_init(60, angle)
    
    # save figure with different names depend on the view
    filename='D:\\Python\\datasets\\3dvideo\\'+str(angle)+'.png'
    plt.savefig(filename, dpi=75)
#%%
from PIL import Image
png_count = 180
files = []
for i in range(png_count):
    seq = str(i*2)
    file_names = 'D:\\Python\\datasets\\3dvideo\\'+ seq +'.png'
    files.append(file_names)    
print(files)
#%%
# Create the frames
frames = []
files
for i in files:
    new_frame = Image.open(i)
    frames.append(new_frame)
    
# Save into a GIF file that loops forever   
frames[0].save('D:\\Python\\datasets\\3dvideo\\3d_vis.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=40, loop=0)
#%%





