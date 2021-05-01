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

rootdir = '../recognizing-faces-in-the-wild/train-faces/'

#returns the family number from a filepath
def returnFamilyNum(string):
    x = re.split("train-faces/F", string)
    secondx=x[1]
    return secondx[0:3]
    

#gets files from all directories
files_in_dir = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print (os.path.join(subdir, file))
        files_in_dir.append(os.path.join(subdir, file))
# print(files_in_dir)


# iterate through the list of filenames
with open('pairs.csv', mode='w+', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Image1', 'Image2', 'Label'])
    for i in files_in_dir:
        for j in files_in_dir:
            firstFacePath = i
            secondFacePath = j
            label = 1 if returnFamilyNum(firstFacePath) == returnFamilyNum(secondFacePath) else 0
            writer.writerow([firstFacePath, secondFacePath, label])
            

fileTest = files_in_dir[1]
image = skimage.io.imread(fileTest, as_gray=True)
#viewer = skimage.viewer.ImageViewer(image)
#viewer.show()

# create the histogram
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))


# configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()










