%% this scprit is used for training 10 autoencoders
function [opttheta]= getVisualFeatures()
    tic;
    datanum = [100,300,500,700,900,1000];
    traintimes = [20,40,60,80,100];
    listlength = size(datanum);
    timeslength = size(traintimes);
    llength = listlength(1,2);
    tlength = timeslength(1,2);
    data = loadMNISTImages('mnist/train-images.idx3-ubyte');
    labels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
    hiddenSize = 200;
    visibleSize = 784;

    wfinal = [];
    for l = 1:llength
        digitsnum = datanum(l)
        dirpath = getdirpath(digitsnum); 
        if exist(dirpath)==7
            fprintf('%s is already exist!\n',dirpath);
        else
            mkdir(dirpath);
        end
        trainingset = gettrainingset(digitsnum,data,labels);
        %permutation examples
        trainingset(:,randperm(size(trainingset,2)));
        filepath = strcat(dirpath,'/traindata.mat');
        fprintf('%s \n',filepath);
        i =1;
        w = [];
        for j = 1:tlength 
            filep= getfilepath(i,dirpath,traintimes(j));
            disp(filep);
            opttheta = trainvisual(i,trainingset,traintimes(j));
            patches = reshape(opttheta(1:hiddenSize*visibleSize),hiddenSize,visibleSize);
            pach  = getpatch(patches,hiddenSize);
            w = [w;pach];
            save(filep,'opttheta');
        end
        wfinal=[wfinal;w];
        save(filepath,'trainingset')
    end
    W = [];
    for i=6:-1:1
        W = [W;wfinal(5*i-4:5*i,:)];
    end
    save('wvisual.mat','W');
    toc;
end    
    
function [filepath] = getfilepath(i,dirpath,times)
    fhead = 'opttheta';
    ftail = '.mat';
    autoid = i;
    t = num2str(times);
    fid = num2str(autoid);
    file = strcat(strcat(fhead,fid),'_');
    file = strcat(file,t);
    file = strcat(file,ftail);
    filepath = strcat(strcat(dirpath,'/'),file);
end

function [dirpath] = getdirpath(j)
    prefix = 'aec_analysis/';
    dhead ='aeccost_'; 
    dirid = 10*j; 
    did = num2str(dirid);
    dirname = strcat(dhead,did);
    dirpath = strcat(prefix,dirname);
end

function [patch] = getpatch(wholeweights,hidden)
    k = randi([1,hidden],1);
    patch = wholeweights(k,:);
end
