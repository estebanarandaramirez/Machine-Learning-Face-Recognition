from operator import indexOf
# from numpy import testing
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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def getLabels():
    labeldir = "test-public-lists/test-public-lists/"
    uniqueLabels = set()
    for subdir, dirs, files in os.walk(labeldir):
        for f in files:
            df = pd.read_csv(labeldir+f)
            for i in range(len(df)):
                fileOne = df.iloc[i]['p1']
                fileTwo = df.iloc[i]['p2']
                uniqueLabels.add(fileOne+fileTwo)
    return uniqueLabels


def testFiles(file_name, svm):
    #the pairs of files to be tested
    df_jpgpairs = pd.read_csv(
        "recognizing-faces-in-the-wild/test-private-lists/test-private-lists/{}.csv".format(file_name))

    #the labels for those pairs
    df_labels = pd.read_csv(
        "recognizing-faces-in-the-wild/test-private-labels/test-private-labels/{}.csv".format(file_name))

    #directory with all of our test faces
    testdir = "recognizing-faces-in-the-wild/test-private-lists/test-private-lists/"

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


#returns the family number from a filepath
def returnFamilyMember(string):
    x = re.split("\\\\", string)
    familyMember = x[-3] + '/' + x[-2] + '/'
    return familyMember


def returnImageName(string):
    return string[(len(string)-9):len(string)]


def appendHistograms(fPOne, fPTwo):
    #read images
    imageOne = skimage.io.imread(fPOne, as_gray=True)
    imageTwo = skimage.io.imread(fPOne, as_gray=True)

    #create histograms
    histogramOne, bin_edgesOne = np.histogram(imageOne, bins=256, range=(0, 1))
    histogramTwo, bin_edgesTwo = np.histogram(imageTwo, bins=256, range=(0, 1))
    
    hist = np.append(histogramOne, histogramTwo)
    return hist


rootdir = 'train'

uniqueLabels = getLabels()

#gets files from all directories
files_in_dir = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        files_in_dir.append(os.path.join(subdir, file))

# iterate through the list of filenames
histograms, labels = np.array([]), np.array([])
for fPOne in files_in_dir[:20]:
    for fPTwo in files_in_dir[:20]:
        hist = appendHistograms(fPOne, fPTwo)
        label = 1 if returnFamilyMember(fPOne) + returnFamilyMember(fPTwo) in uniqueLabels else 0
        if label:
            print(returnFamilyMember(fPOne) + returnFamilyMember(fPTwo))
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
Z = svmLinear.predict(histograms)
print("Accuracy:", np.mean(labels==Z))

# #Printing to a csv
# with open('pairs.csv', mode='w+', newline='') as f:
#     writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['img_pair','is_related'])
#     count=0
#     for fPOne in files_in_dir[:20]:
#         for fPTwo in files_in_dir[:20]:
#             jpg_pair=returnImageName(fPOne)+"-"+returnImageName(fPTwo)
#             writer.writerow([jpg_pair, Z[count]])
#             count=count+1
# f.close()

# # iterate through test-private files
# testdir = "recognizing-faces-in-the-wild/test-private-lists/test-private-lists/"

# for subdir, dirs, files in os.walk(testdir):
#     for f in files:
#         testFiles(f.split('.')[0], svmLinear)

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
