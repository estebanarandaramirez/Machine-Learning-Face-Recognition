from operator import indexOf
# from numpy import testing
import sklearn.svm
import skimage.color
import skimage.io
import skimage.viewer
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import glob
import re
import csv
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

train_labels_fpath = 'recognizing-faces-in-the-wild/train_relationships.csv'
df_trlabels = pd.read_csv(train_labels_fpath)

def testFiles(file_name, svm):
    #the pairs of files to be tested
    df_jpgpairs = pd.read_csv(
        "recognizing-faces-in-the-wild/test-private-lists/test-private-lists/{}.csv".format(file_name))

    #the labels for those pairs
    df_labels = pd.read_csv(
        "recognizing-faces-in-the-wild/test-private-labels/test-private-labels/{}.csv".format(file_name))

    #directory with all of our test faces
    testDir = "recognizing-faces-in-the-wild/test-private-lists/test-private-lists/"

    testHistograms = np.array([])
    testLabels=df_labels['label'].to_numpy()
    for x in range(len(df_jpgpairs)):
        #get paths for two files
        fpOne = testDir+df_jpgpairs.iloc[x]['p1']
        fpTwo = testDir+df_jpgpairs.iloc[x]['p2']

        #return appended histogram for combination of images
        colorData = appendHistograms(fpOne, fpTwo)

        testHistograms = np.vstack((testHistograms, colorData)) if testHistograms.size != 0 else np.array(colorData)

    Z = svm.predict(testHistograms)
    # print(testLabels[:5])
    # print(Z[:5])
    print("Accuracy for {}:".format(file_name), np.mean(testLabels==Z))


#returns the family folder from a filepath
def returnFamilyNum(string):
    return (string[len(string)-27:len(string)-22] +
          "/" + string[len(string)-21:len(string)-17])
    # secondx=x[1]
    # return secondx[0:10]


def returnImageName(string):
    return string[(len(string)-9):len(string)]

def checkIfPairExists(image1, image2):
    global df_trlabels
    # print(type(image1))
    # print(type(df_trlabels['p1'].to_list()))
    p1_list = df_trlabels['p1'].to_list()
    p2_list = df_trlabels['p2'].to_list()
    for x in range(len(p1_list)):
        print(image2,image1)
        if image1==str(p1_list[x]):
            if image2==str(p2_list[x]):
                print(image1,image2,"true")
                return True


    return False

checkIfPairExists("F0002/MID1", "F0002/MID3") #test call

def appendHistograms(fPOne, fPTwo):
    #read images
    imageOne = skimage.io.imread(fPOne, as_gray=True)
    imageTwo = skimage.io.imread(fPOne, as_gray=True)

    #create histograms
    histogramOne, bin_edgesOne = np.histogram(imageOne, bins=256, range=(0, 1))
    histogramTwo, bin_edgesTwo = np.histogram(imageTwo, bins=256, range=(0, 1))
    
    hist = np.append(histogramOne, histogramTwo)
    return hist


rootdir = 'recognizing-faces-in-the-wild/train-faces'

#gets files from all directories
files_in_dir = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print (os.path.join(subdir, file))
        files_in_dir.append(os.path.join(subdir, file))

# iterate through the list of filenames
histograms, labels = np.array([]), np.array([])
for fPOne in files_in_dir[:20]:
    img1 = str(returnFamilyNum(fPOne))
    for fPTwo in files_in_dir[:20]:
        hist = appendHistograms(fPOne, fPTwo)
        img2=str(returnFamilyNum(fPTwo))

        # print(img1,img2)
        returnFamilyNum(fPOne)
        # print(checkIfPairExists(img1, img2))
        label = 1 if checkIfPairExists(img1,img2) else 0
        if histograms.size != 0:
            histograms = np.vstack((histograms, hist))
        else:
            histograms = np.array(hist)
        if labels.size != 0:
            labels = np.hstack((labels, label))
        else:
            labels = np.array(label)

svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
svmLinear.fit(histograms, labels) #Where X is an array of color arrays
print(labels[:20])
Z = svmLinear.predict(histograms)
print(Z[:20])

#Printing to a csv
with open('pairs.csv', mode='w+', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['img_pair','is_related'])
    count=0
    for i, fPOne in enumerate(files_in_dir[:50]):
        for j, fPTwo in enumerate(files_in_dir[:50]):
            jpg_pair=returnImageName(fPOne)+"-"+returnImageName(fPTwo)
            writer.writerow(
                [jpg_pair, Z[count]])
            count=count+1
f.close()

# iterate through test-private files
testdir = "recognizing-faces-in-the-wild/test-private-lists/test-private-lists/"

for subdir, dirs, files in os.walk(testdir):
    for f in files:
        testFiles(f.split('.')[0], svmLinear)

# #fileTest = files_in_dir[1]
# #image = skimage.io.imread(fileTest, as_gray=True)
# #viewer = skimage.viewer.ImageViewer(image)
# #viewer.show()

# # create the histogram
# #histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))


# # configure and draw the histogram figure
# #plt.figure()
# #plt.title("Grayscale Histogram")
# #plt.xlabel("grayscale value")
# #plt.ylabel("pixels")
# #plt.xlim([0.0, 1.0])  # <- named arguments do not work here

# #plt.plot(bin_edges[0:-1], histogram)  # <- or here
# #plt.show()
