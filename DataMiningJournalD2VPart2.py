# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:18:36 2019

Developed for Akdeniz University Graduate School of Natural And Applied Sciences Computer Engineering Department CSE 5010 - DATA MINING Course

Lecturer: Prof. Dr. Melih GÜNAY

@author: Ugur HAZIR and Seth Michail
"""
#import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd
#import multiprocessing
#import gensim.models.word2vec as w2v
import gensim.models.doc2vec as d2v
from gensim.summarization.textcleaner import tokenize_by_word
#from gensim.test.utils import common_texts, get_tmpfile

model = d2v.Doc2Vec.load("doc2vec.model")
docVectors = model.wv
df = pd.read_pickle("df.pkl")
train = pd.read_pickle("train.pkl")
test = pd.read_pickle("test.pkl")
w1 = ["conclusion"]
print(model.wv.most_similar(w1))

nTrain = train.AbstractText.shape[0]
nTest = test.AbstractText.shape[0]

AbstractWordListsTrain = []
Xtrain = np.zeros((nTrain, 700))
for i in range(nTrain):
	# strip problem chars and split into list
    g = list(tokenize_by_word(train.AbstractText.iloc[i].replace("<p>", "").replace("</p>", "")))
    nLengthOfString = len(g)
    AbstractWordListsTrain.append(g)
    for j in range(nLengthOfString):
        if g[j] in docVectors.vocab:
            wordVector = list(model.wv[g[j]])
            for k in range(700):
                Xtrain[i][k] += wordVector[k] # vectorized abstracts

AbstractWordListsTest = []
Xtest = np.zeros((nTest, 700))
for i in range(nTest):
    g = list(tokenize_by_word(test.AbstractText.iloc[i].replace("<p>", "").replace("</p>", "")))
    nLengthOfString = len(g)
    AbstractWordListsTrain.append(g)
    for j in range(nLengthOfString):
        if g[j] in docVectors.vocab:
            wordVector = list(model.wv[g[j]])
            for k in range(700):
                Xtest[i][k] += wordVector[k]
    
print("\r\n", AbstractWordListsTrain[0][0], model.wv[AbstractWordListsTrain[0][0]], len(AbstractWordListsTrain[0]))

print(Xtrain[0])
print(Xtest[0])

dsXtrain = pd.DataFrame(Xtrain)
dsXtrain.to_pickle("Xtrain.pkl")

dsXtest = pd.DataFrame(Xtest)
dsXtest.to_pickle("Xtest.pkl")