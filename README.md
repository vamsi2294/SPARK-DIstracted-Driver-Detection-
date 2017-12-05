Introduction

Dataset
We obtained the dataset from Kaggle State farm competition.
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data
The Distracted Driver dataset consists of images taken for each person within different classes over a set of 22425 color images in 10 classes. 
Data Collection
The dataset used for this project is Distracted driver detection competition dataset from Kaggle. It contains colored images with 640X480 resolution. Image are displayed below.

Data downloaded has  
Training Models
For any image classification Neural networks is suitable. Convolution neural Networks are
Neural Networks:
Neural network uses artificial neurons which are interconnected to different layers which recognize the patterns by calculating the error from the training data which results in better learning rate for the model. The least squared error function is used to train the data to reduce the error based on the weights on each node for the neural network.

 
Input layer: It is input for the model where it learns from the given input data. Each input network is given a weight along with the inputs. The inputs are 50X50 pixels and weights are randomly initialized with matrix size input neurons by number of Hidden neurons.
Hidden layer: These units are in between input and output layers, it’s job is to transform the input features by using the Error functions or weighted functions that are used by the Output layer. 
Output layer: This unit obtains the result from the model and how it has learned from the given structure. For this project the total output neurons are 10 as we must classify the data into 10 categories. 
For this project we have implemented 3-layer neural network with one input layer, 1 hidden layer with 50 nodes and 1 output layer with 10 nodes. In Feed forward propagation each image is trained by taking pixels and calculating the dot product of weights. For forward propagation we have used Sigmoid activation function to predict the output using the random weights in initial phase. We have used Tanh sigmoid function.
a1=xW1+b1
z1=sigmoid(a1)
The activation functions are applied to the intermediate output z1. The weights W1 and W2 transform the data in between the layers. The output predicted form the forward propagation is calculated using the hidden layer weight and the output generated by the hidden layer. 
a2=z1W2+b2
y(pred)=Softmax(a2)
Backward Propagation: 
For backward propagation each output predicted is compared to the training label and the error is calculated and is used in gradient descent algorithm to improve the model.  The loss is used to learn the weights of the model. 
Backward Propagation Formulas
loss=y(pred)-y
δ2=(1-〖tanhz1〗^2 ).(W1.loss)
δW1=xTδ2
δW2=z1xT
The weights are updated by using the formulas:
W1=W1-δW1*Learning Rate
W2=W2-δW2*Learning Rate
The weights obtained with default learning rate of 0.01 and these weights form the model for our test dataset which are predicted with forward propagation. 
KNN:
K-NN classification, the output is a Classified based on the nearest neighbors to the given data point. For k=1 the classifier considers only single nearest data point to determine the class. For K=n point closer to the n data points in the dataset. It is best model to interpret the output, with less calculation times and best prediction power compared to different models. 
Euclidean Distance=√(∑_(i=1)^k▒(x_i-y_j )^2 )
Manhatten Distance= ∑_(i=1)^k▒|x_i ̇ -y_i | 
For each image the distance using Euclidean distance and Manhattan distance for each training image is calculated and determine the nearest neighbors for each test data.

Observations and Evaluation
Neural Networks:
With increase in number of iterations, accuracy is increasing. But after 55 iterations accuracy is reduced. It is overfitting for more than 55 iterations. So, we have limited to 50 iterations. 
With increase in number of images per class, there is an increase in accuracy and increase in run time. Maximum accuracy is achieved at 30 iterations for 2200 images.  For 2200 images at 20 iterations the execution time is around 1 hr 30 min and for 500 images at 40 iterations is 2 hours. At 50 hidden nodes. If we increase the hidden layers to 100 the accuracy is reduced by 10%.  Here is a sample output for the neural Network implementation on Hadoop cluster. 
Output for 30 iterations and 500 images per class
 
K-NN:
K-NN performed best for k=1 and number of images per class are 250 at 80% accuracy. As we increase the k values the accuracy reduced drastically.  Here is a sample output for k=1 and images per class=250.
 
Dependencies:

Instructions
References:

 

