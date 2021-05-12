from operator import indexOf
from numpy import testing
import sklearn.svm
import skimage.color
import skimage.io
# import skimage.viewer
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import glob
import re
import csv
import pandas as pd
import tensorflow as tf
# import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image

#returns the family number from a filepath
def returnFamilyNum(string):
    x = re.split("F", string)
    secondx = x[1]
    return secondx[0:4]


def returnImageName(string):
    return string[(len(string)-9):len(string)]


def appendData(fPOne, fPTwo):
    #read images
    imageOne = skimage.io.imread(fPOne, as_gray=True)
    imageTwo = skimage.io.imread(fPOne, as_gray=True)

    #create histograms
    histogramOne, bin_edgesOne = np.histogram(imageOne, bins=256, range=(0, 1))
    histogramTwo, bin_edgesTwo = np.histogram(imageTwo, bins=256, range=(0, 1))
    hist = np.append(histogramOne, histogramTwo)

    #create image arrays
    imgs = np.vstack((np.array(imageOne), np.array(imageTwo)))

    return hist, imgs


rootdir = 'train'

#gets files from all directories
files_in_dir = []
for subdir, dirs, files in os.walk(rootdir, followlinks=True):
    for file in files:
        files_in_dir.append(os.path.join(subdir, file))

# iterate through the list of filenames
histograms, images, labels = np.array([]), np.array([]), np.array([])

for fPOne in files_in_dir[:30]:
    for fPTwo in files_in_dir[:30]:
        hist, img = appendData(fPOne, fPTwo)
        label = 1 if returnFamilyNum(fPOne) == returnFamilyNum(fPTwo) else 0
        if histograms.size != 0:
            histograms = np.vstack((histograms, hist))
        else:
            histograms = np.array(hist)
        if images.size != 0:
            images = np.vstack((images, img))
        else:
            images = np.array(img)
        if labels.size != 0:
            labels = np.hstack((labels, label))
        else:
            labels = np.array(label)

images = np.reshape(images, (images.shape[0]//448, 448, 224))
svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
svmLinear.fit(histograms, labels)  # Where X is an array of color arrays,
Z = svmLinear.predict(histograms)
print("Accuracy:", np.mean(labels == Z))

# #Printing to a csv
# with open('pairs.csv', mode='w+', newline='') as f:
#     writer = csv.writer(f, delimiter=',', quotechar='"',
#                         quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['img_pair', 'is_related'])
#     count = 0
#     for i, fPOne in enumerate(files_in_dir[:45]):
#         for j, fPTwo in enumerate(files_in_dir[:45]):
#             jpg_pair = returnImageName(fPOne)+"-"+returnImageName(fPTwo)
#             writer.writerow(
#                 [jpg_pair, Z[count]])
#             count = count+1
# f.close()

# test
df_jpgpairs = pd.read_csv("test-private-lists/test-private-lists/ss.csv")
df_labels = pd.read_csv("test-private-labels/test-private-labels/ss.csv")

#directory with all of our test faces
testDir = "test-private-faces/test-private-faces/"

testHistograms, testImages = np.array([]), np.array([])
testLabels = df_labels['label'].to_numpy()
for x in range(1000):
    fpOne = testDir+df_jpgpairs.iloc[x]['p1']
    fpTwo = testDir+df_jpgpairs.iloc[x]['p2']

    hist, img = appendData(fpOne, fpTwo)

    testHistograms = np.vstack((testHistograms, hist)) if testHistograms.size != 0 else np.array(hist)
    testImages = np.vstack((testImages, img)) if testImages.size != 0 else np.array(img)

testImages = np.reshape(testImages, (testImages.shape[0]//448, 448, 224))
Z = svmLinear.predict(testHistograms)
print("Accuracy:", np.mean(testLabels[:1000] == Z[:1000]))

#Deep Learning Method
model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(1, (448, 224)), 
  tf.keras.layers.Flatten(), 
  tf.keras.layers.Dense(128, activation='relu'), 
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(images, labels, epochs=10)
Z = model.predict(testImages)
Z = np.argmax(Z, axis=1)
# print(Z[:500])
# print(testLabels[:500])
print("Accuracy:", np.mean(testLabels[:1000] == Z[:1000]))