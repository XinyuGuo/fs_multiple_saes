function testeachperformance();
   addpath '../deeplearning'
   test = loadMNISTImages('./../mnist/t10k-images.idx3-ubyte');
   testlabels = loadMNISTLabels('./../mnist/t10k-labels.idx1-ubyte');
   testlabels(testlabels ==0 ) = 10;
   load('./aeccost_3000/labels.mat','datalabels');
   load('./aeccost_3000/traindata.mat','trainingset');

   datadir = './aeccost_3000/'; 
   aecnum = 15;    
   datanum = size(testlabels,1);
   truth = [];
   for i = 1:aecnum  
       fprintf('SAE %d\n',i);
       filename = strcat('stackedParas',num2str(i),'.mat');
       filepath = strcat(datadir,filename);
       load(filepath,'stackedAETheta');
       netconfig.inputsize = 784;
       netconfig.layersizes = [{40}];
       [pred] = autoSoftPredict(stackedAETheta,784,40,10,netconfig,test);
      
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
   end
end
