from pyspark import SparkContext

sc = SparkContext(appName="Preprocess")
import sys
import os  
import io
import cv2   
import time
import random
import pandas as pd
import numpy as np 

from random import shuffle  
from matplotlib import pyplot as plt

#Adding color data to the images.
def color(image,bins=(8,8,8)):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    cv2.normalize(hist,hist)
    return(list(hist.flatten()))

#Function to apply several preprocessing functions to the images.
def preprocess(img):
	y=np.fromstring(img, np.uint8)
	img=cv2.imdecode(y,1);
    
    # Constants for finding range of skin color in YCrCb
	min_YCrCb = np.array([0,113,80],np.uint8)
	max_YCrCb = np.array([255,153,125],np.uint8)
	imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image

	skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
	img = cv2.bitwise_and(img,img,mask=skinRegion)
	img_color = color(img)
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img = cv2.resize(img,(50,50))
	img = img.reshape(50*50)
	img = list(img)+img_color
	img = list(img)
	return(img)

start_time = time.time()
print(start_time)
labels = []
data_dir=sys.argv[1]
target_dir=sys.argv[2]

#Looping to preprocess and create csv for all the images in the class.
for i in range(0,10):
	path = data_dir + "/c" + str(i) + "/";
	imagesRDD = sc.binaryFiles(path);
	imageRDD = imagesRDD.map(lambda x:x[1])
	img = imageRDD.map(preprocess)
	x = img.collect()
	dataFrame = pd.DataFrame(x)
	dataFrame['labels'] = "c" + str(i);
	target_path = target_dir + "/c" + str(i) + ".csv";
	dataFrame.to_csv(target_path, sep=',')
	print("c" + str(i) + ": "+str(time.time()-start_time))  
print(time.time()-start_time)
