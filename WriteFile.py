# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 19:26:55 2021

@author: nightrain


Summary

Use the open() function with the w or a mode to open a text file for appending.

Always close the file after completing writing using the close() method or 
use the with statement when opening the file.

Use write() and writelines() methods to write to a text file.

Pass the encoding='utf-8' to the open() function to write UTF-8 characters into a file.


"""
#%%
import glob
"""
The glob module finds all the pathnames matching a specified pattern 
according to the rules used by the Unix shell, 
although results are returned in arbitrary order.
"""
import numpy as np

import pathlib

"""
The pathlib is a Python module which provides an object API for 
working with files and directories. 
"""

#%%
pathsave="D:/Python/report/"
filename=pathsave+'test'
name=['mean','std']
#%%
"""
example1 for creating file
"""
# fp = open(filename+".txt", "a")
# fp.write('start'+'\n')
# fp.close()
#%%
"""
example2 for creating file
"""
with open(filename+".txt", "w") as fp:
    fp.write('Readme')

# 'w'	Open a text file for writing text
#%%
lines = ['Readme', 'How to write text files in Python']
with open(filename+".txt", "a") as fp:
    for line in lines:
        fp.write(line)
        fp.write('\n')

# 'a'	Open a text file for appending text，可以接續前面的內容持續寫入
#%%
lines = ['Readme', 'How to write text files in Python']
with open(filename+".txt", "a") as fp:
    fp.writelines(lines)
    fp.write('\n')
#%%
"""
If you treat each element of the list as a line, 
you need to concatenate it with the newline character like this:
"""
lines = ['Readme', 'How to write text files in Python']
with open(filename+".txt", "a") as fp:
    fp.write('\n'.join(lines))
#%%
"""
map() function returns a map object(which is an iterator) of the results 
after applying the given function to each item of a given iterable (list, tuple etc.)


.join，將前面的字後加入join()中的東西
"""

for i in range(10):
    
    dummydata=np.random.rand(10)
    
    with open(filename+".txt", "a") as fp:
        fp.write('\n'.join(map(str,list(zip(name,[dummydata.mean(),dummydata.std()]))))+'\n'+'\n')

with open(filename+".txt", "a") as fp:
    fp.write('\n'+'finished')
#%%
"""
To open a file and write UTF-8 characters to a file, 
you need to pass the encoding='utf-8' parameter to the open() function.
"""
quote = '成功を収める人とは人が投げてきたレンガでしっかりした基盤を築くことができる人のことである。'

with open(filename+".txt", "a" ,encoding='utf-8') as fp:
    fp.write('\n'+quote)

#%%
"""
save numpy array to text
"""
dummydata=np.random.rand(10)
np.savetxt(filename+'value'+'.txt', dummydata)

# with open(filename+'value'+".txt", "a" ,encoding='utf-8') as fp:
#     fp.write('\n'+quote)

#%%
"""

find the text data

"""
# pathname=(pathlib.Path().absolute())
# pathname=str(pathname)
pathname=pathsave
path=pathname+"/*.txt"
result=glob.glob(path)
for f in result:
    print(f)
#%%
# load text to numpy array 無法讀文字
b = np.loadtxt(f, dtype=float)
#b=b.reshape(len(b),1)
#%%
print(b)

#%%
"""
read text file :
    
Summary

Use the open() function with the 'r' mode to open a text file for reading.

Use the read(), readline(), or readlines() method to read a text file.

Always close a file after completing reading it using the close() method 

or the with statement.

Use the encoding='utf-8' to read the UTF-8 text file.

"""
#%%
"""
read() – read all text from a file into a string. 

This method is useful if you have a small file and 
you want to manipulate the whole text of that file.

readline() – read the text file line by line and return all the lines as strings.

readlines() – read all the lines of the text file and return them as a list of strings.

"""
#%%
with open(pathsave+"test.txt",'r', encoding='utf8') as f:
    contents = f.read()
    print(contents)

# 'r'	Open for text file for reading text
#%%
with open(pathsave+"test.txt",'r', encoding='utf8') as f:
    contents = f.readlines()
for s in contents :
    print(s)
#%%
with open(pathsave+"test.txt",'r', encoding='utf8') as f:
    while True:
        line = f.readline()
        if not line: 
            break
        print(line)      
#%%



