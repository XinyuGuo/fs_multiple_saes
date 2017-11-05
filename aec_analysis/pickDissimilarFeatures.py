from __future__ import division
import scipy.io as importer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import hog
from pylab import *
import time
def getHogFeatures(data):
    numberofimages = np.shape(data)[0]
    d = np.reshape(data,[numberofimages,28,28],order='F')
    digitimage = []
    for i in range(numberofimages):
        digitimage.append(hog(d[i,:,:], orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=True)[0])
    digithogs = np.array(digitimage)
    return digithogs

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def getSAEparas(filelist,vsize,hsize):
    encoder = []
    decoder = []
    bias1 = []
    bias2 = []
    for i in range(len(filelist)):
        mat= importer.loadmat(filelist[i])
        print filelist[i]
        paras = mat['sae1OptTheta']
        W1 = paras[0:hsize*vsize].reshape(hsize,vsize,order='F')
        W2 = paras[hsize*vsize:2*hsize*vsize].reshape(vsize,hsize,order='F')
        b1 = paras[2*hsize*vsize:2*hsize*vsize+hsize]
        b2 = paras[2*hsize*vsize+hsize:2*hsize*vsize+vsize+hsize]
        encoder.append(W1)
        decoder.append(W2)
        bias1.append(b1)
        bias2.append(b2)
    e = np.array(encoder)
    d = np.array(decoder)
    b1 = np.array(bias1)
    b2 = np.array(bias2)
    return e,d,b1,b2

def getSim(hogfeatures):
    numfeatures = hogfeatures.shape[0]
    sim = []
    def getKey(item):
        return item[1]
    for i in range(numfeatures):
        for j in range(i+1,numfeatures):
            edist = np.linalg.norm(hogfeatures[i,:]-hogfeatures[j,:])
            index = np.array([i,j])
            tup = (index,edist)
            sim.append(tup)
    sortedsim = sorted(sim,key=getKey,reverse=True)
    return sortedsim

def checkSetsim(featureset,tup,hogfeatures):
    chosenf = hogfeatures[featureset,:]
    index = tup[0]
    intoset = False
    count = 0
    hog0 = hogfeatures[index[0],:]
    hog1 = hogfeatures[index[1],:]
    for i in range(len(chosenf)):
        edist1 = np.linalg.norm(hog0-hogfeatures[featureset[i],:])
        edist2 = np.linalg.norm(hog1-hogfeatures[featureset[i],:])
        if edist1> 2.5 and edist2 >2.5:
           count = count + 1
    if count is len(chosenf):
        intoset = True
    return intoset

def pickFeatures(similarity,features,hogfeatures,bias,fnum):
    N = len(similarity)
    i = 0
    j = 0
    fset = []
    while (i<N) and (j<fnum):
       tup=similarity[i]
       index = tup[0]
       print tup[1]
       if index[0] not in fset and index[1] not in fset:
           if checkSetsim(fset,tup,hogfeatures):
               fset.append(index[0])
               fset.append(index[1])
               j=j+2
       i=i+1
    featureset =features[fset,:]
    biasset = bias[fset]
    print fset
    return featureset,biasset

start_time = time.time()
filepaths = ['./aeccost_1000/','./aeccost_3000/','./aeccost_5000/','/aeccost_7000','/aeccost_9000','./aeccost_10000']
traintimes = ['60/','70/','80/','90/','100/']
visibleSize = 784
hiddenSize = 200
filepathnum = len(filepaths)
traintimesnum = len(traintimes)

for i in range(filepathnum):
    prefix = filepaths[i]
    for j in range(traintimesnum):
        directory = traintimes[j]
        numofsae = 10
        fnamelist = []
        for i in range(1,numofsae+1):
            filename = 'opt'+str(i)+'.mat'
            fname = prefix+directory+filename
            fnamelist.append(fname)

        encoder,decoder,bias1,bias2 = getSAEparas(fnamelist,visibleSize,hiddenSize)

        l = range(numofsae)
        data = np.concatenate((encoder[l,:,:]),axis=0)
        data_d = np.concatenate((decoder[l,:,:]),axis=1)
        data_b = np.concatenate(bias1[l,:,:])
        #print data_b.shape
        #b1 = np.mean(bias1,axis=0)
        #b1 = b1[0:50]
        #fpath = './aeccost_10000/features.mat'
        ofeaturepath=prefix+traintimes[j]+'originalfeatures.mat'
        importer.savemat(ofeaturepath,{'ofeatures':data})
        #fp = './aeccost_10000/decoder.mat'
        #importer.savemat(fp,{'decoder':data_d})
        #print encoder.shape
        #print decoder.shape
        #print bias1.shape
        #print bias2.shape

        hogfeatures = getHogFeatures(data) #400*128
        #print hogfeatures.shape
        sim = getSim(hogfeatures)
        #print len(sim)
        #print sim
        featureset,b1 = pickFeatures(sim,data,hogfeatures,data_b,200)
        #print featureset.shape
        #print b1.shape
        featurename ='features'
        biasname = 'b1'

        fpath = prefix+traintimes[j]+'features.mat'#'./aeccost_10000/features.mat'
        bpath = prefix+traintimes[j]+'b1.mat'#'./aeccost_10000/b1.mat'

        importer.savemat(fpath,{featurename:featureset})
        importer.savemat(bpath,{biasname:b1})
print("--- %s seconds ---" % (time.time() - start_time))
