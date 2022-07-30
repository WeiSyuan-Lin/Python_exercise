# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 20:27:25 2022

@author: nightrain
"""


from rake_nltk import Rake
import nltk
nltk.download('punkt')
#Rapid Automatic Keyword Extraction
#%%
rake_nltk_var = Rake()
#%%
text = """
The process of extracting keywords helps us identifying the importance of words in a text. This task can be also used for topic modelling. 
It is very useful to extract keywords for indexing the articles on the web so that people searching the keywords can get the best articles to read.
"""
#%%
rake_nltk_var.extract_keywords_from_text(text)
keyword_extracted = rake_nltk_var.get_ranked_phrases()
print(keyword_extracted)