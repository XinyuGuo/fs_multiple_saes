from __future__ import division
import scipy.io as importer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import hog
from pylab import *

def getHogFeatures(data):
    numberofimages = np.shape(data)[0]
    d = np.reshape(data,[numberofimages,28,28])
    digitimage = []
    for i in range(numberofimages):
        digitimage.append(hog(d[i,:,:], orientations=8, pixels_per_cell=(7, 7),cells_per_block=(1, 1), visualise=True)[0])
    digithogs = np.array(digitimage)
    return digithogs

filepath = './aeccost_30000/weights.mat'
data = importer.loadmat(filepath)
data = data['weights']
hogfeatures = getHogFeatures(data)
path = './aeccost_30000/featurehog.mat'
importer.savemat(path,{'featurehog':hogfeatures})
