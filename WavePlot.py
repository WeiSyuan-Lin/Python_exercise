# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:32:01 2022

@author: nightrain
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import argparse
#%%
if __name__== "__main__":
    parser = argparse.ArgumentParser(description='wave plot')
    parser.add_argument('--f', type=int,default=1,help='frequency')
    parser.add_argument('--l', type=int,default=1,help='length')
    args = parser.parse_args()
    x=np.linspace(-args.l,args.l,200*args.l+1)
    plt.plot(x,np.sin(args.f*np.pi*x))
    plt.show()
#%%
print('the code is finished')