function deeTrain(number,file,parameters)
   [train,datalabels,train2,datalabels2]=gettrainData(number,1000);
    
    %train= loadMNISTImages('mnist/train-images.idx3-ubyte'); %784*60000); %784*60000
    %datalabels= loadMNISTLabels('mnist/train-labels.idx1-ubyte');
    %datalabels(datalabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
    test = loadMNISTImages('mnist/t10k-images.idx3-ubyte'); %784*60000
    testlabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
    testlabels(testlabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
   %disp(size(train));
   %disp(size(datalabels));
   %disp(size(train2));
   %disp(size(datalabels2));
   kt = false;
   disp('no knowledge');
   m = false;
   stackedAEKT([],[],train,datalabels,test,testlabels,kt,m,file,parameters);
   % 30000 data 
   disp('with knowledge');
   m = true;
   stackedAEKT([],[],train,datalabels,test,testlabels,kt,m,file,parameters);
