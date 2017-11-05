import scipy.io as importer
import numpy as np
from sklearn.cluster import AgglomerativeClustering

filepath_1 = './aeccost_30000/aec1.mat'
filepath_2 = './aeccost_30000/aec2.mat'
filepath_3 = './aeccost_30000/aec3.mat'

data_1 = importer.loadmat(filepath_1)
data_1 = data_1['w']
data_2 = importer.loadmat(filepath_2)
data_2 = data_2['w']
data_3 = importer.loadmat(filepath_3)
data_3 = data_3['w']

data_12 = np.concatenate((data_1,data_2),axis = 0)
data = np.concatenate((data_12,data_3),axis = 0)

print data.shape

#print data.shape
n_clusters = 5

ward = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward').fit(data)
labels = ward.labels_
y = np.bincount(labels)
#print y
#print np.nonzero(y)[0]
ii = np.nonzero(y)[0]
groupdis = zip(ii,y[ii])
print groupdis

label= np.where(labels == 4)[0]
print label[(0<=label)&(label<200)].size
print label[(200<=label)&(label<400)].size
print label[(400<=label)&(label<600)].size

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

groups = [None]*n_clusters
for clusterid in range(n_clusters):
    #indices = indices(labels,lambda id: id == clusterid)
    #ind = np.array(indices)
    #groups[clusterid] = data[ind]
    groups[clusterid] = data[np.array(indices(labels,lambda id:id==clusterid))]
    print groups[clusterid].shape
    filepath = './aeccost_30000/group%d.mat' % clusterid
    print filepath
    groupname ='arr%d' % clusterid
    importer.savemat(filepath,{groupname:groups[clusterid]})
#print np.count_nonzero(ward.labels_)
#print type(ward.labels_)

