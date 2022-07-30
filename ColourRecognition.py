# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 23:33:46 2022

@author: nightrain



failed

"""

import numpy as np
import pandas as pd
import cv2
#%%
path_img='D:\\Python\\datasets\\colour\\colour.jpeg'
#%%
img = cv2.imread(path_img)
#%%
path='https://raw.githubusercontent.com/amankharwal/Website-data/master/colors.csv'
#%%
index=["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(path, names=index, header=None)
#%%
clicked = False
r = g = b = xpos = ypos = 0
#%%
def recognize_color(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        # if(d&lt;=minimum):
        #     minimum = d
        cname = csv.loc[i,"color_name"]
    return cname
#%%
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)
#%%
cv2.imshow("Color Recognition App",img)
#%%
# cv2.namedwindow('color recognition app')
# cv2.setmousecallback('color recognition app', mouse_click)
# while(1):
#     cv2.imshow("color recognition app",img)
#     if (clicked):
#         cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)
#         text = recognize_color(r,g,b) + ' r='+ str(r) +  ' g='+ str(g) +  ' b='+ str(b)
#         cv2.puttext(img, text,(50,50),2,0.8,(255,255,255),2,cv2.line_aa)
#         if(r+g+b>=600):
#             cv2.puttext(img, text,(50,50),2,0.8,(0,0,0),2,cv2.line_aa)
#         clicked=false
#     if cv2.waitkey(20) & 0xff ==27:
#         break
# cv2.destroyallwindows()











