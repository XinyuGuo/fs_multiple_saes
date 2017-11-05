function featuresoft()
   load('../aec_analysis/aeccost_3000/traindata.mat','trainingset');
   load('../aec_analysis/aeccost_3000/labels.mat','datalabels');
   test = loadMNISTImages('mnist/t10k-images.idx3-ubyte'); %784*60000
   testlabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
   testlabels(testlabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

   b = true;
   vsize = 784;
   hsize = 200;
   numofClasses = 10;
   theta1 = initializeParameters(hsize,vsize); 
   saeSoftmaxTheta = 0.005*randn(hsize*numofClasses,1);
   paras = cell(1,2);
   paras{1,1} = theta1;
   paras{1,2} = saeSoftmaxTheta;

   [stackedAETheta] = trainfeaturesoft(hsize,vsize,numofClasses,trainingset,datalabels,test,testlabels,paras);
   netconfig.inputsize = 784;
   netconfig.layersizes = [{200}];
   [pred] = autoSoftPredict(stackedAETheta,784,200,10,netconfig,test);
   truthtable = testlabels(:)==pred(:);

   one = find(truthtable==1);
   allright = testlabels(one); 
   for i = 1 :10
      num = size(find(allright==i),1);
      NUM = size(find(testlabels==i),1);
      fprintf('the total number of digit %d is %d\n',i,NUM);
      fprintf('the number of digit %d is %d\n',i,num);
      fprintf('the percentage is %0.3f%%\n',num/NUM*100);
   end

   fprintf('\n');

   zero = find(truthtable==0);
   allwrong = testlabels(zero); 
   size(allwrong)
   for i = 1 :10
      num = size(find(allwrong==i),1);
      NUM = size(find(testlabels==i),1);
      fprintf('the total number of digit %d is %d\n',i,NUM);
      fprintf('the number of digit %d is %d\n',i,num);
      fprintf('the percentage is %0.3f%%\n',num/NUM*100);
   end

   beforeTunepred = mean(testlabels(:)==pred(:));
   fprintf('Test: Before Finetuning Test Accuracy %0.3f%%\n',beforeTunepred*100);

end
