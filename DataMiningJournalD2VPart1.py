# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:05:47 2019

Developed for Akdeniz University Graduate School of Natural And Applied Sciences Computer Engineering Department CSE 5010 - DATA MINING Course

Lecturer: Prof. Dr. Melih GÜNAY

@author: Ugur HAZIR and Seth Michail

"""

import pyodbc
import tensorflow as tf
import numpy as np # linear algebra
import matplotlib.pyplot as plt #Plotting
import pandas as pd
import multiprocessing
import gensim.models.word2vec as w2v
import gensim.models.doc2vec as d2v
from gensim.summarization.textcleaner import tokenize_by_word
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#commented out----------------------------------------------------------------
"""
conn_str=(
    r"Driver={SQL Server Native Client 11.0};"
    r"Server=10.1.128.249,1433;"#write local instead of ip for local connection
    r"Database=CompSciencePub;"
    r"Trusted_Connection=no;"
    r"uid=compscience;"
    r"pwd=CompScience2005;"
    )

conn_str2=(
    r"Driver={SQL Server Native Client 11.0};"
    r"Server=(local);"
    r"Database=CompSciencePub;"
    r"Trusted_Connection=no;"
    r"uid=saDataMining;"
    r"pwd=Pp123456;"
    )

conn = pyodbc.connect(driver='{SQL Server Native Client 11.0}',
                               server='local',
                               database='[DATAMINING]',
                               uid='saDataMining',pwd='Pp123456')
"""
#this is new, copied(with modification) from https://datatofish.com/how-to-
#connect-python-to-sql-server-using-pyodbc/-----------------------------------
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-C2V3H1M\SQLEXPRESS;'
                      'Database=DATAMINING;'#try changing CompSciencePub to DATAMINING
                      'Trusted_Connection=yes;')

cursor = conn.cursor()
"""cursor.execute('SELECT * FROM db_name.Table')"""
#-----------------------------------------------------------------------------

#commented out----------------------------------------------------------------
#conn=pyodbc.connect(conn_str)

#cursor = conn.cursor()
#cursor.execute('SELECT TOP 5 * FROM [dbo].[Table1]')
#cursor.execute('SELECT TOP 5 * FROM [CompSciPublications].[AcademicRecord]')
#-----------------------------------------------------------------------------

#for use with Database=DATAMINING in conn declaration
query = """ SELECT * FROM [dbo].[Table4] """

#commented out----------------------------------------------------------------
#for use with Database=CompSciencePub in conn declaration
#query="""WITH tbl as
#                    (
#                    SELECT t1.AcademicRecordID,t1.PublicationId,t1.Title,t3.Name,t3.Abbreviation,t2.AbstractText
#                      FROM [CompSciPublications].[AcademicRecord] as t1
#
#                    	INNER JOIN [CompSciPublications].[AcademicRecordAbstract] as t2
#                    		ON t1.AcademicRecordID=t2.AcademicRecordId
#
#                   	INNER JOIN [CompSciPublications].[Publication] as t3
#                  		ON t1.PublicationId=t3.PublicationID
#                  )
#                    SELECT *
#                    FROM tbl"""
#"""db=cursor.execute(WITH tbl as
#                    (
#                    SELECT t1.AcademicRecordID,t1.PublicationId,t1.Title,t3.Name,t3.Abbreviation,t2.AbstractText
#                      FROM [CompSciPublications].[AcademicRecord] as t1
#                    	INNER JOIN [CompSciPublications].[AcademicRecordAbstract] as t2
#                    		ON t1.AcademicRecordID=t2.AcademicRecordId
#                    	INNER JOIN [CompSciPublications].[Publication] as t3
#                    		ON t1.PublicationId=t3.PublicationID
#                    )
#                    SELECT TOP 5 *
#                    FROM tbl)
#df = pd.DataFrame(db.fetchall())
#df.columns = db.keys()"""
#-----------------------------------------------------------------------------

df = pd.read_sql(query, conn)

print("\r\nHead:\r\n")
head = df.head() # Gives first 5 rows
print(head)

print("\r\nTail:\r\n")
tail = df.tail() # Gives last 5 rows
print(tail)

print("\r\n")
print(df.columns)
print("\r\n")
print(df.info())
print("\r\n")
#df.describe not applicable
print(df.describe())

publications = (np.sort(df.PublicationID.unique()).tolist())#was there a reason for 'Id' instead of 'ID' here and elsewhere?

#print("\r\n Publications:")
#print(publications)
nPublications = len(publications)
print("\r\n number of unique publications: ", nPublications, type(publications))
n = df.AbstractText.shape[0] # number of abstracts in dataframe
print("\r\n Number of Abstracts: ", n)
msk = np.random.rand(len(df)) < 0.75
train = df[msk].copy() # training set of abstracts
test = df[~msk].copy() # testing set of abstracts

nTrain = train.AbstractText.shape[0] # number of abstracts in training set, should be msk*n
nTest = test.AbstractText.shape[0] # number of abstracts in test set, should be 1-msk*n
print("\r\n Number of Train Abstracts: ",nTrain)
print("\r\n Number of Test Abstracts: ",nTest)

Ytrain = np.zeros((nTrain, nPublications))
Ytest = np.zeros((nTest, nPublications))

trainPublicationsList = list(train['PublicationID'])
testPublicationsList = list(test['PublicationID'])

for i in range(nTrain):
    for j in range(nPublications):
        if(trainPublicationsList[i] == publications[j]):
            Ytrain[i][j] = 1 # abstract j in pubs is abstract i in train, 
            break            # Ytrain is a matrix to map these indices
        
for i in range(nTest):
    for j in range(nPublications):
        if(testPublicationsList[i] == publications[j]):
            Ytest[i][j] = 1
            break

#print("\r\n Words split test:")

#df["AbstractText2"]=df.AbstractText.replace("<p>","").replace("</p>","")

AbstractWordListsTrain = []
for i in range(nTrain):
    g = list(tokenize_by_word(train.AbstractText.iloc[i].replace("<p>", "").replace("</p>", "")))
    AbstractWordListsTrain.append(g)

#print(g, type(g))

cores = multiprocessing.cpu_count()

#sentences=[g]
print("\r\nCores: ", cores)

print("\r\n Word vectora are calculating please wait for a while")
#create a list of 'tagged' documents, used by gensim
documents = [TaggedDocument(AbstractWordListsTrain[i], [i]) for i, 
             AbstractWordListsTrain[i] in enumerate(AbstractWordListsTrain)]
#train the model, using the given tagged document list, to generate vector 
#representations of the tagged documents, or something like that(not a loop)
'''
#seed (int, optional), #seed should not be used, see documentation
dm = 1; distributed memory
dm = 0; distributed bag-of-words
dm_concat = 0; use dm_mean (optional)
dm_concat = 1; use concatentation (HUGE model)
dm_tag_count (int, optional); used with dm_concat 
dm_mean = 0; use sum of context word vectors
dm_mean = 1; use avg of context word vectors
dm_concat and dm_mean give horrible results
dbow_words = 0; train doc vecs
dbow_words = 1; train doc vecs + skip-gram, simultaneously
ns_exponent = 1.0; sample exactly by frequency
ns_exponent = 0.0; sample all words equally (default is 0.75)
ns_exponent < 0.0; sample low freq more than high freq
hs = 0; if also 'negative' > 0, negative sampling used
hs = 1; heirarchical softmax
epochs (int, optional, default = 5); num of iterations over corpus (list of tagged docs)
vector_size (int, optional, default = 300); number of features 
alpha (0.0 <= a <= 1.0, optional); learning rate
min_alpha (float, optional) – Learning rate will linearly drop to min_alpha as training progresses
min_vocab_size (int, optional); set to "None" for no pruning
min_count (int, optional); ignores words that occur fewer than this number of times
sample (0.0 to 1E-5, optional); threshold for config of random downsample of high freq words
window (int, optional) – The maximum distance between the current and predicted word within a sentence
workers (int, optional) – Use these many worker threads to train the model (=faster training with multicore machines)
'''
model = Doc2Vec(documents,
				dm = 0,
				vector_size = 700,
				window = 3,
				alpha = 0.06,
				#min_alpha = 0.025,
                min_count = 3,
				max_vocab_size = None,
				sample = 0.001,
				workers = 8,
				epochs = 15,
				hs = 0,
                negative = 7,
				ns_exponent = -0.7,
				dbow_words = 0
				)
'''
should look for work by others to decide which hparams to adjust at the same time
hyperparams to try: min_count in [2,10], vector_size in [100,1000], window in [1,10],

'''
#d2v.Doc2Vec()
print("\r\nEpochs: ", model.epochs)                
#model.build_vocab(sentences)
#model.train(sentences)
path = get_tmpfile("doc2vec.model")
model.save("doc2vec.model")
df.to_pickle("df.pkl")
train.to_pickle("train.pkl")
test.to_pickle("test.pkl")

dsYtrain = pd.DataFrame(Ytrain)
dsYtrain.to_pickle("Ytrain.pkl")

dsYtest = pd.DataFrame(Ytest)
dsYtest.to_pickle("Ytest.pkl")

dsPublications=pd.DataFrame(publications)
dsPublications.to_pickle("Publications.pkl")

####### PART 2 starts here *******************

