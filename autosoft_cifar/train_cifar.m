function train_cifar()
  load('cifar-10-batches-mat/trainlabels.mat','trainlabels');
  load('cifar-10-batches-mat/train.mat','training');
  load('cifar-10-batches-mat/test.mat','datatest');
  load('cifar-10-batches-mat/testlabels.mat','labelstest');
  labelstest(labelstest==0)= 10;
  trainlabels(trainlabels==0) = 10;
  b = false;
  theta1 = initializeParameters(400,3072);
  saeSoftmaxTheta = 0.005*randn(400*10,1);
  paras = cell(1,2);
  paras{1,1} = theta1;
  paras{1,2} = saeSoftmaxTheta;
  autono = 1000;
  softno = 100;
  autoSoft(training',trainlabels,datatest',labelstest,b,paras,autono,softno,1);
end
