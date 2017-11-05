function trainautoSoft(number,autono,softno,i)
   %[train,datalabels,train2,datalabels2]=gettrainData(number,1);
   %save('../aec_analysis/aeccost_10000/traindata.mat','train');
   %save('../aec_analysis/aeccost_10000/labels.mat','datalabels');

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   load('../aec_analysis/aeccost_3000/traindata.mat','trainingset');
   load('../aec_analysis/aeccost_3000/labels.mat','datalabels');

   %load('../aec_analysis/aeccost_3000/trainwrong.mat','datawrong');
   %load('../aec_analysis/aeccost_3000/wrongdatalabels.mat','wrongdatalabels');
   test = loadMNISTImages('mnist/t10k-images.idx3-ubyte'); %784*60000
   testlabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
   testlabels(testlabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
   %test = train2;
   %testlabels = datalabels2;
   %load ('~/Dropbox/autoencoder_study/aec_analysis/aeccost_10000/traindata.mat','trainingset')
   b = true;

   theta1 = initializeParameters(40,784); 
   saeSoftmaxTheta = 0.005*randn(40*10,1);
   paras = cell(1,2);
   paras{1,1} = theta1;
   paras{1,2} = saeSoftmaxTheta;
   
   autoSoft(trainingset,datalabels,test,testlabels,b,paras,autono,softno,i);
