%% this scprit is used for training 10 autoencoders  
function [opttheta]= trainaecs()
    data = loadMNISTImages('mnist/train-images.idx3-ubyte');
    labels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
    digitsnum = 100;
    trainingset = gettrainingset(digitsnum,data,labels);
    %permutation examples
    trainingset(:,randperm(size(trainingset,2)));
    save('traindata.mat','trainingset')
    disp(size(trainingset)) % debug
    %error('exit!')

    aec_num = 1;
    for i =1:aec_num
        filename = getfilename(i);
        disp(filename);
        opttheta = train(i,trainingset);
        save(filename,'opttheta');
    end
    
    function [file] = getfilename(i)
        fhead = 'opttheta';
        ftail = '.mat';
        autoid = i;
        fid = num2str(autoid);
        file = strcat(strcat(fhead,fid),ftail);
       
