from operator import indexOf
from numpy import testing
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

rootdir = 'recognizing-faces-in-the-wild/train-faces'

#returns the family number from a filepath
def returnFamilyNum(string):
    x = re.split("F", string)
    secondx=x[1]
    return secondx[0:4]

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
    
#gets files from all directories
files_in_dir = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print (os.path.join(subdir, file))
        files_in_dir.append(os.path.join(subdir, file))


# iterate through the list of filenames
histograms, labels = np.array([]), np.array([])
for fPOne in files_in_dir[:20]:
    for fPTwo in files_in_dir[:20]:
        hist = appendHistograms(fPOne, fPTwo)
        label = 1 if returnFamilyNum(fPOne) == returnFamilyNum(fPTwo) else 0
        if histograms.size != 0:
            histograms = np.vstack((histograms, hist))
        else:
            histograms = np.array(hist)
        if labels.size != 0:
            labels = np.hstack((labels, label))
        else:
            labels = np.array(label)

# print(histograms.shape, labels.shape)
svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
svmLinear.fit(histograms, labels) #Where X is an array of color arrays,
Z = svmLinear.predict(histograms)
# print(histograms)
# print(labels)
# print(Z)     
# print(type(Z))

#Printing to a csv
with open('pairs.csv', mode='w+', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['img_pair','is_related'])
    count=0
    for i, fPOne in enumerate(files_in_dir[:20]):
        for j, fPTwo in enumerate(files_in_dir[:20]):
            jpg_pair=returnImageName(fPOne)+"-"+returnImageName(fPTwo)
            writer.writerow(
                [jpg_pair, Z[count]])
            count=count+1
f.close()

#test one, testing the pairs and labels in the corresponding bb.csv fies

#the pairs of files to be tested
df_jpgpairs = pd.read_csv(
    "recognizing-faces-in-the-wild/test-private-lists/test-private-lists/bb.csv")

#the labels for those pairs
df_labels = pd.read_csv(
    "recognizing-faces-in-the-wild/test-private-labels/test-private-labels/bb.csv")

#get labels in an array, same format as our prediction output
testLabels=df_labels['label'].to_numpy()
# testLabels = testLabels[]

# print(type(testLabels))

#directory with all of our test faces
testDir = "recognizing-faces-in-the-wild/test-private-faces/test-private-faces/"

testHistograms = np.array([])

for x in range(len(df_jpgpairs)):
    #get paths for two files
    fpOne = testDir+df_jpgpairs.iloc[x]['p1']
    fpTwo = testDir+df_jpgpairs.iloc[x]['p2']

    #return appended histogram for combination of images
    colorData = appendHistograms(fpOne, fpTwo)

    testHistograms = np.vstack((testHistograms, colorData)) if testHistograms.size != 0 else np.array(colorData)


Z = svmLinear.predict(testHistograms)
print(testLabels.shape)
print(Z.shape)
# print(testLabels[:5])
# print(Z[:5])
print("Accuracy:", np.mean(testLabels==Z))

# def (filename1, filename2,  index):
#     filepath = "recognizing-faces-in-the-wild/test-private-faces/test-private-faces"

#     fpath1=filepath+filename1
#     fpath2=filepath+filename2
#     hist = appendHistograms(fpath1,fpath2)









# iterate through the list of filenames
# with open('pairs.csv', mode='w+', newline='') as f:
#     writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(['Image1', 'Image2', 'Histogram1', 'Histogram2', 'Label', 'Combined_Histogram'])
#     for fPOne in files_in_dir[:10]:
#         for fPTwo in files_in_dir[:10]:
#             # hist=[]
#             histogramOne, histogramTwo = appendHistograms(fPOne, fPTwo)
#             hist = np.append(histogramOne, histogramTwo)
#             # X.append(hist)
#             # print(X)
#             # hist.extend(histogramTwo)
#             label = 1 if returnFamilyNum(fPOne) == returnFamilyNum(fPTwo) else 0
#             writer.writerow([fPOne, fPTwo, histogramOne, histogramTwo, label, hist])
# f.close()

# df = pd.read_csv('pairs.csv')
# df_header = ['Image1', 'Image2', 'Histogram1',
#              'Histogram2', 'Label', 'Combined_Histogram']

# data=df[['Combined_Histogram']].values
# # np.genfromtxt()
# labels = df[['Label']].values


# label_encoder = LabelEncoder()

# labels = label_encoder.fit_transform(labels)

# data = label_encoder.fit_transform(data)

# data = data.reshape(-1, 1)

# # print(hist)



# #data = StandardScaler().fit_transform(data)




# # print(data)
# svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
# svmLinear.fit(data,labels)


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











