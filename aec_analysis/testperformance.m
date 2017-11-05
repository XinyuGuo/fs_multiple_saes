function testperformance()
   %addpath '../';% loadMNISTImages() and loadMNISTLabels() directory
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
   truth2 =[];

   for i = 1:aecnum  
       filename = strcat('stackedParas',num2str(i),'.mat');
       filepath = strcat(datadir,filename);
       load(filepath,'stackedAETheta');
       netconfig.inputsize = 784;
       netconfig.layersizes = [{200}];
       [pred] = autoSoftPredict(stackedAETheta,784,200,10,netconfig,test);
       [pred2] = autoSoftPredict(stackedAETheta,784,200,10,netconfig,trainingset);
      
       truthtable = testlabels(:)==pred(:);
       truthtable2 = datalabels(:) ==pred2(:);

       beforeTunepred = mean(truthtable);%mean(testlabels(:)==pred(:)); 
       beforeTunepred2 = mean(truthtable2);%mean(testlabels(:)==pred(:)); 
       fprintf('Test: Before Finetuning Test Accuracy %0.3f%%\n',beforeTunepred*100);
       fprintf('Training: Before Finetuning Test Accuracy %0.3f%%\n',beforeTunepred2*100);
       truth = [truth,truthtable];
       truth2 = [truth2,truthtable2];
   end

   data = sum(truth,2)./15;
   binranges = [0,0.2,0.4,0.6,0.8,1];
   [bincounts] = histc(data,binranges);
   bincounts

   data2 = sum(truth2,2)./15;
   binranges2 = [0,0.2,0.4,0.6,0.8,1];
   [bincounts2] = histc(data2,binranges2);
   bincounts2
   %figure
   %bar(binranges,bincounts,'histc');
    
   one = find(data==1);
   allright = testlabels(one); 
   
   for i = 1 :10
      num = size(find(allright==i),1);
      NUM = size(find(testlabels==i),1);
      fprintf('the total number of digit %d is %d\n',i,NUM);
      fprintf('the number of digit %d is %d\n',i,num);
      fprintf('the percentage is %0.3f%%\n',num/NUM*100);
   end

   fprintf('\n');

   zero = find(data==0);
   allwrong = testlabels(zero); 

   size(allwrong)

   for i = 1 :10
      num = size(find(allwrong==i),1);
      NUM = size(find(testlabels==i),1);
      fprintf('the total number of digit %d is %d\n',i,NUM);
      fprintf('the number of digit %d is %d\n',i,num);
      fprintf('the percentage is %0.3f%%\n',num/NUM*100);
   end
    
  
   one2 = find(data2==1);
   allright2 = testlabels(one2); 
   
   for i = 1 :10
      num = size(find(allright2==i),1);
      NUM = size(find(datalabels==i),1);
      fprintf('Training :the total number of digit %d is %d\n',i,NUM);
      fprintf('Training :the number of digit %d is %d\n',i,num);
      fprintf('Training :the percentage is %0.3f%%\n',num/NUM*100);
   end

   fprintf('\n');
   zero2 = find(data2==0);
   datawrong = trainingset(:,zero2);
   save('./aeccost_3000/trainwrong.mat','datawrong');
   allwrong2 = testlabels(zero2); 
   wrongdatalabels = allwrong2;
   save('./aeccost_3000/wrongdatalabels.mat','wrongdatalabels');
   size(allwrong2)

   for i = 1 :10
      num = size(find(allwrong2==i),1);
      NUM = size(find(datalabels==i),1);
      fprintf('Training :the total number of digit %d is %d\n',i,NUM);
      fprintf('Training :the number of digit %d is %d\n',i,num);
      fprintf('Training :the percentage is %0.3f%%\n',num/NUM*100);
   end
  % [R,P]=corrcoef(truth');  
  % colormap('hot');             
  % imagesc(R);
  % colorbar;
end
