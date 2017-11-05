from __future__ import division
import scipy.io as importer
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import hog
from pylab import *

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
        if edist1> 3.0 and edist2 >3.0:
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

numofsae = 10
fnamelist = []
for i in range(1,numofsae+1):
    filename = 'opt'+str(i)+'.mat'
    fname = './aeccost_10000/'+filename
    fnamelist.append(fname)

visibleSize = 784
hiddenSize = 200
encoder,decoder,bias1,bias2 = getSAEparas(fnamelist,visibleSize,hiddenSize)

l = range(numofsae)
data = np.concatenate((encoder[l,:,:]),axis=0)
data_d = np.concatenate((decoder[l,:,:]),axis=1)
data_b = np.concatenate(bias1[l,:,:])
#print data_b.shape
#b1 = np.mean(bias1,axis=0)
#b1 = b1[0:50]
fpath = './aeccost_10000/features.mat'
importer.savemat(fpath,{'features':data})

#fp = './aeccost_10000/decoder.mat'
#importer.savemat(fp,{'decoder':data_d})
print encoder.shape
print decoder.shape
print bias1.shape
print bias2.shape

hogfeatures = getHogFeatures(data) #400*128
print hogfeatures.shape
sim = getSim(hogfeatures)
#print len(sim)
#print sim
featureset,b1 = pickFeatures(sim,data,hogfeatures,data_b,200)
print featureset.shape
print b1.shape
featurename ='features'
biasname = 'b1'

fpath = './aeccost_10000/features.mat'
bpath = './aeccost_10000/b1.mat'

importer.savemat(fpath,{featurename:featureset})
importer.savemat(bpath,{biasname:b1})
"""
n_clusters =10

ward = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(hogfeatures)
labels = ward.labels_
y = np.bincount(labels)
#print y
#print np.nonzero(y)[0]
ii = np.nonzero(y)[0]
groupdis = zip(ii,y[ii])
print groupdis

'''test - how many features comes from each aec for a certain group'''
#label= np.where(labels == 4)[0]
#print label[(0<=label)&(label<200)].size
#print label[(200<=label)&(label<400)].size
#print label[(400<=label)&(label<600)].size

groups = [None]*n_clusters
hfeatures = [None]*n_clusters
for clusterid in range(n_clusters):
    #indices = indices(labels,lambda id: id == clusterid)
    #ind = np.array(indices)
    #groups[clusterid] = data[ind]
    groups[clusterid] = data[np.array(indices(labels,lambda id:id==clusterid))]
    hfeatures[clusterid] = hogfeatures[np.array(indices(labels,lambda id:id==clusterid))]
    print groups[clusterid].shape
    filepath = './aeccost_10000/group%d.mat' % clusterid
    print filepath
    groupname ='arr%d' % clusterid
    #importer.savemat(filepath,{groupname:groups[clusterid]})
#print np.count_nonzero(ward.labels_)
#print type(ward.labels_)
#grouphf = np.array(hfeatures)
rate = 100/200
groupindice = []
for i in range(n_clusters):
    absolutevalue = np.sqrt(np.sum(np.square(hfeatures[i]),1))
    indice = np.argsort(absolutevalue)
    groupindice.append(indice)

gindice = np.array(groupindice)
all = 0
features =[]
for j in range (n_clusters):
    num = np.shape(gindice[j])[0]
    features.append(groups[j][gindice[j][range(0,num,2)],:])
    all = all+np.shape (gindice[j][range(0,num,2)])[0]

print all

rawfeatures = np.array(features)
#print np.shape(rawfeatures[0])
aa = rawfeatures[0]
for k in range(1,n_clusters):
    aa = np.append(aa,rawfeatures[k],axis=0)
print aa.shape
if aa.shape[0]>200:
    extra = aa.shape[0]-200
    positions = range(extra)
    print positions
    a = np.delete(aa,positions,0)
else:
    a = aa
print a.shape

featurename ='features'
biasname = 'b1'
fpath = './aeccost_10000/features.mat'
bpath = './aeccost_10000/b1.mat'
importer.savemat(fpath,{featurename:a})
importer.savemat(bpath,{biasname:b1})
"""
