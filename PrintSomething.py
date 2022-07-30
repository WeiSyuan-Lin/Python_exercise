# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:25:48 2022

範例: 直接執行 python script 並加入參數
使用 if __name__== "__main__": 執行冒號下方的主程式
argparse.ArgumentParser  收集主程式需要的參數

@author: nightrain
"""
#%%
"""
The argparse module makes it easy to write user-friendly command-line interfaces. 
The program defines what arguments it requires

The argparse module also automatically generates help and usage messages 
and issues errors when users give the program invalid arguments.


"""
#%%
"""
Every Python module has it’s __name__ defined and 
if this is ‘__main__’, it implies that the module is being run standalone by the user 
and we can do corresponding appropriate actions.

If you import this script as a module in another script, 
the __name__ is set to the name of the script/module.

Python files can act as either reusable modules, or as standalone programs.

if __name__ == “main”: is used to execute some code only if the file was run directly, 

and not imported.
"""
#%%
import numpy as np
import argparse

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='test input')
    parser.add_argument('--N', type=int,default=10,help='number of length')
    parser.add_argument('--C', type=float,default=None,help='number of factor')
    args = parser.parse_args()
    
    #print("please enter a number")
    #N=input()
    a=args.C*np.ones(int(args.N))
    print(a)

