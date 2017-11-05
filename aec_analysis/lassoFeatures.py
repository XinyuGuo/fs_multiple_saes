from sklearn import linear_model
import scipy.io as importer
import numpy as np
import numpy.matlib
import math
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
'''
concatenate 15 SAEs' weights together
'''
def getRepresentations(data):
    filepath = './aeccost_3000/backup_200/'
    file_prefix ='stackedParas'
    file_extension = '.mat'
    weights = []
    bias = []
    for i in range(1,16,1):
        filename = file_prefix + str(i) + file_extension
        thisfile = filepath + filename
        mat = importer.loadmat(thisfile)
        stackedAETheta = mat['stackedAETheta']
        thisweight =  stackedAETheta[2000:2000+784*200].reshape(200,784,order='F')
        thisbias = stackedAETheta[2000+784*200:2000+784*200+200]
        weights.append(thisweight)
        bias.append(thisbias)
        npweights = np.array(weights)
        npbias = np.array(bias)
    numofsae = 15
    l = range(numofsae)
    final_weights = np.concatenate(npweights[l,:,:],axis=0)
    final_bias = np.concatenate(npbias[l,:,:],axis=0)
    activation = 1 /(1+np.exp(-np.dot(final_weights,data)-np.transpose(np.matlib.repmat(final_bias,1,data.shape[1]))))
    return activation,final_weights,final_bias

filepath_train = './aeccost_3000/traindata.mat'
filepath_labels= './aeccost_3000/labels.mat'
t_mat = importer.loadmat(filepath_train)
l_mat = importer.loadmat(filepath_labels)
training = t_mat['trainingset']
labels = l_mat['datalabels']
newtraining,features,bias = getRepresentations(training)
print newtraining.shape
train = np.transpose(newtraining)
lsvc = LinearSVC(C=0.01592,penalty="l1",dual=False).fit(train,labels)
model = SelectFromModel(lsvc,prefit=True)
print model.get_support(True)
feature_indices = model.get_support(True)
select_features = features[feature_indices,:]
select_bias = bias[feature_indices]

print select_features.shape
print select_bias.shape
X_new =model.transform(train)
print X_new.shape

tdata = np.transpose(X_new)
print tdata.shape
feature_path = './aeccost_3000/backup_200/lassofeatures.mat'
bias_path= './aeccost_3000/backup_200/lassobias.mat'
importer.savemat(feature_path,{'lassofeatures':select_features})
importer.savemat(biaspe = [n_features]_path,{'lassobias':select_bias})
