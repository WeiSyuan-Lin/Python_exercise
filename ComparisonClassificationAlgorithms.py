# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:07:50 2022

@author: nightrain
"""

import numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
#%%
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/social.csv")
print(data.head())
#%%
x = np.array(data[["Age", "EstimatedSalary"]])
y = np.array(data[["Purchased"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

decisiontree = DecisionTreeClassifier()
logisticregression = LogisticRegression()
knearestclassifier = KNeighborsClassifier()
svm_classifier = SVC()
bernoulli_naiveBayes = BernoulliNB()
passiveAggressive = PassiveAggressiveClassifier()

knearestclassifier.fit(xtrain, ytrain)
decisiontree.fit(xtrain, ytrain)
logisticregression.fit(xtrain, ytrain)
passiveAggressive.fit(xtrain, ytrain)
svm_classifier.fit(xtrain, ytrain.flatten())


data1 = {"Classification Algorithms": ["KNN Classifier", "Decision Tree Classifier", 
                                       "Logistic Regression", "Passive Aggressive Classifier","Support Vector Classifier "],
      "Score": [knearestclassifier.score(xtest,ytest), decisiontree.score(xtest,ytest), 
                logisticregression.score(xtest,ytest), passiveAggressive.score(xtest,ytest),
                svm_classifier.score(xtest,ytest)]}
score = pd.DataFrame(data1)
score
#%%



