#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np # linear algebra
import matplotlib.pyplot as plt #Plotting
import pandas as pd
#from gensim.test.utils import common_texts, get_tmpfile

train = pd.read_pickle("train.pkl")
test = pd.read_pickle("test.pkl")

dsXtrain = pd.read_pickle("Xtrain.pkl")
dsYtrain = pd.read_pickle("Ytrain.pkl")
Xtrain = dsXtrain.values
Ytrain = dsYtrain.values

dsXtest = pd.read_pickle("Xtest.pkl")
dsYtest = pd.read_pickle("Ytest.pkl")
Xtest = dsXtest.values
Ytest = dsYtest.values

dsPublications  = pd.read_pickle("Publications.pkl")
publications = list(dsPublications[0])

trainPublicationsList = list(train['PublicationID'])
testPublicationsList = list(test['PublicationID'])

print("train:", train.shape, "test:", test.shape)
print("Xtrain:", Xtrain.shape, "Ytrain:", Ytrain.shape)
print("Xtest:", Xtest.shape, "Ytest:", Ytest.shape)
print("publications:", len(publications))

nXtrain = Xtrain.shape[0]
nXtest = Xtest.shape[0]
nPublications = len(publications)
#x originally 300, the default for Gensim
x = tf.placeholder(tf.float32, [None, 700])
y_true = tf.placeholder(tf.float32, [None, nPublications])
pkeep = tf.placeholder(tf.float32)

#originally 4 layers, with nodes 1250, 1000, and 500
layer_1 = 1400
layer_2 = 700
layer_3 = 1000
layer_4 = 500
layer_5 = 800
layer_out = nPublications
#try bias decrease instaed of increase, or bias about 25
weight_1 = tf.Variable(tf.truncated_normal([700, layer_1], stddev = 0.025))
bias_1 = tf.Variable(tf.constant(0.01, shape = [layer_1]))
weight_2 = tf.Variable(tf.truncated_normal([layer_1, layer_2], stddev = 0.25))
bias_2 = tf.Variable(tf.constant(0.03, shape = [layer_2]))
weight_3 = tf.Variable(tf.truncated_normal([layer_2, layer_3], stddev = 0.1))
bias_3 = tf.Variable(tf.constant(0.01, shape = [layer_3]))
#originally stddev and constant factor were 0.1 everywhere
#weight_4 = tf.Variable(tf.truncated_normal([layer_3, layer_out], stddev = 0.1))
#bias_4 = tf.Variable(tf.constant(0.1, shape = [layer_out]))
weight_4 = tf.Variable(tf.truncated_normal([layer_3, layer_4], stddev = 0.1))
bias_4 = tf.Variable(tf.constant(0.01, shape = [layer_4]))
weight_5 = tf.Variable(tf.truncated_normal([layer_4, layer_5], stddev = 0.1))
bias_5 = tf.Variable(tf.constant(0.0625, shape = [layer_5]))
weight_6 = tf.Variable(tf.truncated_normal([layer_5, layer_out], stddev = 0.1))
bias_6 = tf.Variable(tf.constant(0.16, shape = [layer_out]))

y1 = tf.nn.sigmoid(tf.matmul(x, weight_1) + bias_1)
y1d = tf.nn.dropout(y1, pkeep)
print('weight 1 shape ',weight_1.shape)
print('weight 6 shape ',weight_6.shape)
print('y1d shape ',y1d.shape)
print('y1 shape ',y1.shape)

y2 = tf.nn.relu(tf.matmul(y1d, weight_2) + bias_2)
y2d = tf.nn.dropout(y2, pkeep)
y3 = tf.nn.relu(tf.matmul(y2d, weight_3) + bias_3)
y3d = tf.nn.dropout(y3, pkeep)
#logits = tf.matmul(y3d, weight_4) + bias_4
#y4 = tf.nn.softmax(logits)
y4 = tf.nn.relu(tf.matmul(y3d, weight_4) + bias_4)
y4d = tf.nn.dropout(y4, pkeep)
y5 = tf.nn.relu(tf.matmul(y4d, weight_5) + bias_5)
y5d = tf.nn.dropout(y5, pkeep)
logits = tf.matmul(y5d, weight_6) + bias_6
print('logits shape ',logits.shape)
y6 = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_true)
loss = tf.reduce_mean(cross_entropy)
#cost = tf.reduce_mean(tf.square(y_true - logits))  # calculates the cost
#accuracy = tf.reduce_mean(tf.cast(cost, tf.float32))
correct_prediction = tf.equal(tf.argmax(y6, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#AdamOptimizer originally set to 0.001
optimize = tf.train.AdamOptimizer(0.00085).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_graph = []
train_step = 500
display_step = 100
def training_step (iterations):
    for i in range (iterations):
		#pkeep originally set to 0.75
        feed_dict_train = {x: Xtrain, y_true: Ytrain, pkeep: 0.82}
        [_, train_loss] = sess.run([optimize, loss], feed_dict = feed_dict_train)
        loss_graph.append(train_loss)

        if i % display_step == 0:
            train_acc = sess.run(accuracy, feed_dict = feed_dict_train)
            """sess.as_default()
            yPredictionTrain=tf.argmax(y4,1)
            predictionTrain = sess.run([yPredictionTrain],feed_dict={x: Xtrain, pkeep: 1})"""
            
            print('Iteration:', i, 'Training accuracy:', train_acc, 'Training loss:', train_loss)


feed_dict_test = {x: Xtest, y_true: Ytest, pkeep: 1}
def test_accuracy ():
    feed_dict_test = {x: Xtest, y_true: Ytest, pkeep: 1}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)

training_step(train_step)
test_accuracy()

fh = open("reportTest.csv", "w")
sess.as_default()
#yPredictionTest = tf.argmax(y4,1)
yPredictionTest = tf.sigmoid(logits)
predictionTest = sess.run([yPredictionTest],feed_dict = {x: Xtest, pkeep: 1})
#print(predictionTest, type(predictionTest), len(predictionTest), len(predictionTest[0]), np.shape(predictionTest))
print("\r\n******************\r\n")
nTestCorrect = 0
for j in range(nXtest):
    isCorrect = 0
    cPrediction = list(predictionTest[0][j])
    #print(cPrediction,type(cPrediction), len(cPrediction))
    npCPredictions = np.zeros((nPublications, 2))
    for i in range(nPublications):
        npCPredictions[i][0] = i
        npCPredictions[i][1] = cPrediction[i]
    npCPredictions = npCPredictions[npCPredictions[:, 1].argsort()[::-1]]
    #print(npCPredictions, type(npCPredictions), npCPredictions.shape)
    predictedIndex = [0,0,0,0,0]
    for i in range(5):
        #print(int(npCPredictions[i][0]))
        predictedIndex[i] = int(npCPredictions[i][0])
        if(testPublicationsList[j] == publications[predictedIndex[i]]):
            if(isCorrect == 0):
                nTestCorrect += 1
            isCorrect = 1
    if j != 0:
        fh.write("\n")
    #fh.write(test.AcademicRecordID.iloc[0] + ',' + testPublicationsList[0] + ',' + publications[predictedIndex] + ',' + isCorrect)
    fh.write(str(test.AcademicRecordID.iloc[j]) + ',' + str(testPublicationsList[j])
        + ',' + str(publications[predictedIndex[0]])
        + ',' + str(publications[predictedIndex[1]])
        + ',' + str(publications[predictedIndex[2]])
        + ',' + str(publications[predictedIndex[3]])
        + ',' + str(publications[predictedIndex[4]])
        + ',' + str(isCorrect))
print("Real accuracy for test:", (nTestCorrect/nXtest)*100)
#print(predictionTest[0][0])
#probTest, predictionTest = sess.run(tf.nn.top_k(yPredictionTest, k=5), feed_dict={x: Xtest[0,], pkeep: 1})
#predictionTest = sess.run([yPredictionTest, yPredictionTest],feed_dict={x: Xtest, pkeep: 1})
#predictionTest = sess.run([yPredictionTest],feed_dict={x: Xtest, pkeep: 1})
#print(predictionTest,type(predictionTest))
#print(predictionTest[0][0])#,predictionTest[1][0])
#print(len(predictionTest))
#fh.write("\r\nIterations: "+str(it)+" Training accuracy: "+str(train_acc)+" Training loss: "+str(train_loss)+" Testing accuracy: "+str(acc))
"""
nTestCorrect=0
for i in range(nXtest):
    isCorrect=0
    predictedIndex=predictionTest[0][i]
    if(testPublicationsList[i]==publications[predictedIndex]):
        isCorrect=1
        nTestCorrect+=1
    if i!=0:
        fh.write("\r\n")
    #fh.write(test.AcademicRecordID.iloc[0]+','+testPublicationsList[0]+','+publications[predictedIndex]+','+isCorrect)
    fh.write(str(test.AcademicRecordID.iloc[i])+','+str(testPublicationsList[i])+','+str(publications[predictedIndex])+','+str(isCorrect))
fh.close()
print("Real accuracy for test:",(nTestCorrect/nXtest)*100)
#print(probTest,type(probTest))
"""
plt.plot(loss_graph, 'k-')
plt.title('Loss graph')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
#moved out of the way from lines 64 to 74 =====================================
#y4 = tf.nn.sigmoid(logits)
"""
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimize = tf.train.AdamOptimizer(0.001).minimize(loss)
"""
#==============================================================================