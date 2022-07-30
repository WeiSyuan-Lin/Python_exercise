# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 20:36:57 2022

@author: nightrain


The main goal of using machine learning for text summarization is 
to reduce the reference text to a smaller version 
while keeping its knowledge alongside its meaning.

"""
#%%
import nltk
import string
from heapq import nlargest
#%%
text = " summary of your desired text that you need to store in the variable ‘text’. I hope you liked this article on Text Summarization with Python. Feel free to ask your valuable questions in the comments section below."
#%%
if text.count(". ") > 20:
    length = int(round(text.count(". ")/10, 0))
else:
    length = 1

nopuch =[char for char in text if char not in string.punctuation]
nopuch = "".join(nopuch)

processed_text = [word for word in nopuch.split() if word.lower() not in nltk.corpus.stopwords.words('english')]

word_freq = {}
for word in processed_text:
    if word not in word_freq:
        word_freq[word] = 1
    else:
        word_freq[word] = word_freq[word] + 1

max_freq = max(word_freq.values())
for word in word_freq.keys():
    word_freq[word] = (word_freq[word]/max_freq)

sent_list = nltk.sent_tokenize(text)
sent_score = {}
for sent in sent_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_freq.keys():
            if sent not in sent_score.keys():
                sent_score[sent] = word_freq[word]
            else:
                sent_score[sent] = sent_score[sent] + word_freq[word]

summary_sents = nlargest(length, sent_score, key=sent_score.get)
summary = " ".join(summary_sents)
#%%
print(summary)