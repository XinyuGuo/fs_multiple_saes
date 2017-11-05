function trainShallowFramework() 
    tic;
    datanum = [100,300,500,700,900,1000];
    traintimes =[60,70,80,90,100];
    test = loadMNISTImages('mnist/t10k-images.idx3-ubyte'); %784*60000
    testlabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
    testlabels(testlabels==0) = 10;
    trainDataPaths = getDataPath();
    datanumlength= size(datanum);
    traintimelength= size(traintimes);
    lend = datanumlength(1,2);
    lent= traintimelength(1,2);

    b = false;
    hiddenSize = 200;
    visibleSize= 784;
    theta1 = initializeParameters(hiddenSize,visibleSize); 
    saeSoftmaxTheta = 0.005*randn(hiddenSize*10,1);
    paras = cell(1,2);
    paras{1,1} = theta1;
    paras{1,2} = saeSoftmaxTheta;

    softno = 10; 
    all_acc = cell(1,lend);
    for i=1:lend
       traindatapaths = getDataPath();
       traindatapath = strcat(traindatapaths{i,1},'/','traindata.mat');
       labelpath = strcat(traindatapaths{i,1},'/','labels.mat'); 
       fprintf('datapath is %s\n',traindatapath);
       fprintf('labelpath is %s\n',labelpath);
       %disp(traindatapath);
       load(traindatapath,'trainingset'); 
       load(labelpath,'datalabels');
       single_auto_acc = [];
       for j=1:lent
           autono =  traintimes(1,j);
           datasavepath =strcat(strcat(traindatapaths{i,1},'/'),num2str(traintimes(1,j)));
           fprintf('files are saved to %s\n',datasavepath);
           if exist(datasavepath)~=7
               mkdir(datasavepath);
           end
           single_auto_acc_each= [];
           for k=1:10
               fileid = num2str(k);
               filename = strcat('opt',fileid,'.mat');
               filepath = strcat(datasavepath,'/',filename);
               [sae1OptTheta,acc] = shallowFramework(trainingset,datalabels,test,testlabels,b,paras,autono,softno,visibleSize,hiddenSize);
               single_auto_acc_each = [single_auto_acc_each,acc];   
               save(filepath,'sae1OptTheta');
               fprintf('save .mat file to %s\n',filepath);
           end
           single_auto_acc = [single_auto_acc;single_auto_acc_each];
       end
       all_acc{1,i} = single_auto_acc;
    end
    save('accuracy.mat','all_acc');
    toc;
end

function [datapaths]= getDataPath()
  datapaths = cell(6,1);
  path1 ='~/projects/autoencoder_study/aec_analysis/aeccost_1000'; 
  path2 ='~/projects/autoencoder_study/aec_analysis/aeccost_3000'; 
  path3 ='~/projects/autoencoder_study/aec_analysis/aeccost_5000'; 
  path4 ='~/projects/autoencoder_study/aec_analysis/aeccost_7000'; 
  path5 ='~/projects/autoencoder_study/aec_analysis/aeccost_9000'; 
  path6 ='~/projects/autoencoder_study/aec_analysis/aeccost_10000'; 
  datapaths{1,1} = path1; 
  datapaths{2,1} = path2;
  datapaths{3,1} = path3;
  datapaths{4,1} = path4;
  datapaths{5,1} = path5;
  datapaths{6,1} = path6;
end

function [path] =getFilePath(datasize)
            
end
