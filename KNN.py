from pyspark import SparkContext

sc = SparkContext(appName="KNN")

import sys
import os  
import io
import cv2   
from PIL import Image
import time
from math import *
from functools import reduce
import random
import pandas as pd
import numpy as np 
from collections import Counter

from random import shuffle

start_time = time.time()

# Funtion to convert the string values in the data to float values
def convert(x):
    y=0
    for i in x:
        x[y] = float(i)
        y+=1
    return x

# Function to find the most common values in the list
def most_common(lst):
    data = Counter(lst)
    return(data.most_common(1)[0][0])

# Function to find the distance between two images
def dist(main,test):
    dis = list(set(zip(main[0:-1],test[0:-1])))
    labels = list(set(zip([main[-1]],[test[-1]])))
    distance = list(map(lambda x:(x[1]-x[0])**2,dis))
    return([sqrt(reduce(lambda x,y:x+y,distance)),labels[0]])

# Function to find the manhattan distance between two images
def mandist(main,test):
    dis = list(set(zip(main[0:-1],test[0:-1])))
    labels = list(set(zip([main[-1]],[test[-1]])))
    distance = list(map(lambda x:abs(x[1]-x[0]),dis))
    return([(reduce(lambda x,y:x+y,distance)),labels[0]])

classes_folder = sys.argv[1]  			#Getting input location from arguments
images_per_class = int(sys.argv[2])		#Getting images per class value from arguments

testRDDs = [];
trainRDDs = [];

# Looping to filter, map and store input data in RDD's
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

test_images = testRDD.count();
train_images = trainRDD.count();


#Finding image classification accuracy
k = int(sys.argv[3])

flag = 0 

#Finding distance of each test image with all train images
for i in testRDD.collect():
    knn = trainRDD.map(lambda x:dist(x,i)).sortByKey(True)
    x = knn.map(lambda y:y[1])
    if(flag==0):
        # knnRDD = sc.parallelize([x.take(k)])
        knnList = [x.take(k)]
    else:
        # knnRDD = knnRDD.union(sc.parallelize([x.take(k)])) 
        knnList = knnList+[x.take(k)]
    flag+=1

knnRDD = sc.parallelize(knnList)
knnRDD = knnRDD.map(lambda x:most_common(x))
print("Matched images: " + str(knnRDD.collect()[0:10]))
matched_images = knnRDD.filter(lambda x:x[0]==x[1]).count();
acc = float(matched_images)/float(test_images);

print("Images per class: " + str(images_per_class))
print("Value of k: " + str(k))
print("Number of train images: " + str(test_images))
print("Number of test images: " + str(train_images))
print("Number of matched classes, total test image count: " + str(matched_images) + "," + str(test_images))
print("accuracy: " + str(acc))

stop_time = time.time()
print("Start time:" + str(start_time))
print("Stop time:" + str(stop_time))
print("Time taken:" + str(stop_time - start_time))