# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:18:20 2022

@author: nightrain
"""

from textblob import TextBlob
words = ["Data Scence", "Mahine Learnin"]
corrected_words = []
for i in words:
    corrected_words.append(TextBlob(i))
print("Wrong words :", words)
print("Corrected Words are :")
for i in corrected_words:
    print(i.correct(), end=" ")