function getDataset()
  load('cifar-10-batches-mat/data_batch_1.mat','data','labels');
  data1 = im2double(data); 
  label1 = labels;
  load('cifar-10-batches-mat/data_batch_2.mat','data','labels');
  data2 = im2double(data); 
  label2 = labels;
  load('cifar-10-batches-mat/data_batch_3.mat','data','labels');
  data3 = im2double(data); 
  label3 = labels;
  load('cifar-10-batches-mat/data_batch_4.mat','data','labels');
  data4 = im2double(data); 
  label4 = labels;
  load('cifar-10-batches-mat/data_batch_5.mat','data','labels');
  data5 = im2double(data); 
  label5 = labels;

  training = [data1;data2;data3;data4;data5]; 
  trainlabels = [label1;label2;label3;label4;label5];
  save('cifar-10-batches-mat/train.mat','training');
  save('cifar-10-batches-mat/trainlabels.mat','trainlabels');

  load('cifar-10-batches-mat/test_batch.mat','data','labels');
  datatest = im2double(data);
  labelstest = labels;
  save('cifar-10-batches-mat/test.mat','datatest');
  save('cifar-10-batches-mat/testlabels.mat','labelstest');
end

