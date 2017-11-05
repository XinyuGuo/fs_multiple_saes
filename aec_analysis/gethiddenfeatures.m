function gethiddenfeatures()
    addpath '../';
    num = 10;
    prefix = 'aeccost_30000/'; 
    sname = 'opttheta';
    dname = 'feature';
    sfile = strcat(prefix,sname); 
    dfile = strcat(prefix,dname); 
    extension = '.mat';
    load('aeccost_30000/traindata.mat');%load training data
    hiddenSize = 200;
    visibleSize = 784;
    for i = 1:num
        fileid = num2str(i);
        spath = strcat(sfile,fileid); 
        dpath = strcat(dfile,fileid); 
        sourcefile = strcat(spath,extension);
        desfile = strcat(dpath,extension);
        load(sourcefile);
        feature = feedForwardAutoencoder(opttheta,hiddenSize,visibleSize,trainingset);
        save(desfile,'feature');
    end
