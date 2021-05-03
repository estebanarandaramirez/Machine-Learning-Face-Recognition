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

rootdir = 'recognizing-faces-in-the-wild/train-faces'

#returns the family number from a filepath
def returnFamilyNum(string):
    # print(string)
    x = re.split("F", string)
    secondx=x[1]
    return secondx[0:4]
    

def appendHistograms(fPOne, fPTwo):
    #read images
    imageOne = skimage.io.imread(fPOne, as_gray=True)
    imageTwo = skimage.io.imread(fPOne, as_gray=True)

    #create histograms
    histogramOne, bin_edgesOne = np.histogram(imageOne, bins=256, range=(0, 1))
    histogramTwo, bin_edgesTwo = np.histogram(imageTwo, bins=256, range=(0, 1))

    return histogramOne, histogramTwo
    
#gets files from all directories
files_in_dir = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print (os.path.join(subdir, file))
        files_in_dir.append(os.path.join(subdir, file))



# iterate through the list of filenames
with open('pairs.csv', mode='w+', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Image1', 'Image2', 'Histogram1', 'Histogram2', 'Label', 'Combined_Histogram'])
    for fPOne in files_in_dir[:30]:
        for fPTwo in files_in_dir[:30]:
            # hist=[]
            histogramOne, histogramTwo = appendHistograms(fPOne, fPTwo)
            hist = np.append(histogramOne, histogramTwo)
            # hist.extend(histogramTwo)
            label = 1 if returnFamilyNum(fPOne) == returnFamilyNum(fPTwo) else 0
            writer.writerow([fPOne, fPTwo, histogramOne, histogramTwo, label, hist])
f.close()

df = pd.read_csv('pairs.csv')
data=df['Combined_Histogram']
# print(data)
svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
svmLinear.fit(data,df['Label'])

            

#fileTest = files_in_dir[1]
#image = skimage.io.imread(fileTest, as_gray=True)
#viewer = skimage.viewer.ImageViewer(image)
#viewer.show()

# create the histogram
#histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))




# configure and draw the histogram figure
#plt.figure()
#plt.title("Grayscale Histogram")
#plt.xlabel("grayscale value")
#plt.ylabel("pixels")
#plt.xlim([0.0, 1.0])  # <- named arguments do not work here

#plt.plot(bin_edges[0:-1], histogram)  # <- or here
#plt.show()











