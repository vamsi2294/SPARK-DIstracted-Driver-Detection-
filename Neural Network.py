
# Package imports
import pyspark
import random
import sys
import time
import numpy as np
from operator import add

# Initializing SparkContext
sc = pyspark.SparkContext(appName="Neural")

start_time = time.time()
# Input arguements
classes_folder=sys.argv[1]
itr=int(sys.argv[2])
images_per_class=int(sys.argv[3])
nn_hdim=int(sys.argv[4])

# Function to convert data to float
def convert(x):
    y=0
    for i in x:
        x[y] = float(i)
        y+=1
    return x

trainRDDs = []
testRDDs = []

# Read the training data from the csv files in HDFS
for i in range(0,10):
	test, train = sc.parallelize(sc.textFile(classes_folder + "/c" + str(i) + ".csv").take(images_per_class)).filter(lambda x:x[0]!=",").map(lambda x:x.split(",")[1:]).map(convert).randomSplit(weights=[0.2, 0.8], seed=1)
	testRDDs.append(test)
	trainRDDs.append(train)

trainRDD = trainRDDs[0];
testRDD = testRDDs[0];

#Looping to union RDD of each class to a single RDD
for i in range(1,10):
	trainRDD = trainRDD.union(trainRDDs[i]);
	testRDD = testRDD.union(testRDDs[i]);


# Intitializing Model weights for the neural Network

def initialize_weights(dim_input, dim_hidden,dim_output):
    np.random.seed(0)
    # initializing the weights based on the input, hidden and output dimensions
    w1 = np.random.randn(dim_input, dim_hidden) / np.sqrt(dim_input)
    w0 = np.random.randn(dim_hidden, dim_output) / np.sqrt(dim_hidden)
    return w1,w0


# Sigmoid Function calcualion

def sigmoid(x):
    return np.tanh(x)

def softmax(z):
    return (np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True))

def forward(W1,W2,X):
    # Creating the Column matrix for the pixels for each image
    a1 = np.column_stack(( X))
    
    # Hidden Layer activation functions
    z1 = a1.dot(W1)
    a1 = sigmoid(z1)

    # Output Layer activation functions
    z2 = a1.dot(W2)
    probs = softmax(z2)
    
    return a1,probs

def predict(model, x):
    W1, W2= model['w1'], model['w2']
    # Forward propagation
    _,probs=forward(W1,W2,x)
    return np.argmax(probs,axis=1)
 

def BackPropagation(W1,W2,x,y):
   
    # Forward porpagation calculation
    a1,delta3= forward(W1,W2,x)
    
    # Back Propagation        
    delta3[0][y] -= 1
    dW2 = (a1.T).dot(delta3)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    # Reshaping the matrix to calculate the dot product
    m=np.reshape(np.array(x),(len(x),1))
    dW1 = np.dot(m, delta2)
    return np.array([dW1,dW2])


#Neural Network Model to train the dataset
def update_weights(w1,w2,average_eval,Lambda,m):
	dw1 = average_eval[0]
	dw2 = average_eval[1]
     

    # Calculating the new weights using the error calculated from the back propagation
	for i in range(w1.shape[0]):
		w1[i, 0] = w1[i, 0] - dw1[i, 0]
	for i in range(w1.shape[0]):
		for j in range(1, w1.shape[1]):
			w1[i, j] = w1[i, j] - (dw1[i, j] + (Lambda/m) * w1[i, j])
	for i in range(w2.shape[0]):
		w2[i, 0] = w2[i, 0] - dw2[i, 0]
	for i in range(w2.shape[0]):
		for j in range(1, w2.shape[1]):
			w2[i, j] = w2[i, j] - (dw2[i, j] + (Lambda/m) * w2[i, j])	

	return w1,w2

def Network(w1,w2, train_set, max_iter):
    
    Lambda=1.1
    # Total number of images used for training
    m = train_set.count() 
    for ik in range(max_iter):
        
        # Mapping each image to train the network
        eval_res = train_set.map(lambda x: BackPropagation(w1,w2,x[0:-1],x[-1]))
        
        # Obtain the results from backpropagation
        average_eval = eval_res.reduce(add) / train_set.count()
        w1,w2=update_weights(w1,w2,average_eval,Lambda,m)
        
    model = { 'w1': w1, 'w2': w2}
    return model

# Input layers and output layers are fixed to 3012 and 10 
input_nodes = 3012
output_nodes = 10 
w1,w2=initialize_weights(input_nodes, nn_hdim,output_nodes)
#gradient decent
model=Network(w1,w2,trainRDD,itr)

# Predicting the Outputs for the test date by using the build Model
y_pred=testRDD.map(lambda x:predict(model,x[0:-1]))
y_test=testRDD.map(lambda x:x[-1])

# Combining to RDD's to predict the output 
y=y_pred.zip(y_test)

# Function to calculate accuracy
def accuracy(x):
	if(x[0]==x[1]):
		return (1)
	else:
		return (0)
# Calling Accuracy function for predicted data
y_acc=y.map(accuracy)
acc=float(y_acc.filter(lambda x: x==1).count())
den=float(y_acc.count())
Imc=trainRDD.count()
Imcc=images_per_class
test=testRDD.count()

# Print Results
print("Number of Training Images :"+str(Imc))
print("Number of Training Images per class:"+str(Imcc))
print("Number of Testing Images:"+str(test))
print("Number of Iterations:"+str(itr))
print("Accuracy:"+str(float(acc/den)))
stop_time=time.time()
print("Time taken:" + str(stop_time - start_time))
sc.stop()