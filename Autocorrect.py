# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 21:38:08 2022

@author: nightrain
"""

path='https://raw.githubusercontent.com/amankharwal/Website-data/master/book.txt'
#%%
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter
words = []
with open(path, 'r',encoding="utf-8") as f:
    file_name_data = f.read()
    file_name_data=file_name_data.lower()
    words = re.findall('\w+',file_name_data)
# This is our vocabulary
V = set(words)
print(f"The first ten words in the text are: \n{words[0:10]}")
print(f"There are {len(V)} unique words in the vocabulary.")
#%%
word_freq_dict = {}  
word_freq_dict = Counter(words)
print(word_freq_dict.most_common()[0:10])
#%%
#Relative Frequency of words
probs = {}     
Total = sum(word_freq_dict.values())    
for k in word_freq_dict.keys():
    probs[k] = word_freq_dict[k]/Total
#%%
# Finding Similar Words
def my_autocorrect(input_word):
    input_word = input_word.lower()
    if input_word in V:
        return('Your word seems to be correct')
    else:
        similarities = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index':'Word', 0:'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
        return(output)
#%%
my_autocorrect('morningg')
#%%



