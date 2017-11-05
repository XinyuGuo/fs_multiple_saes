function getDataandLabels()
    prefix = '~/projects/autoencoder_study/aec_analysis/';
    folders = {'aeccost_1000','aeccost_3000','aeccost_5000','aeccost_7000','aeccost_9000','aeccost_10000'};
    numfolders = size(folders,2); 
    datanum = [100,300,500,700,900,1000];
    for i=1:numfolders
        datapath = strcat(prefix,folders{1,i},'/','traindata.mat');
        labelpath = strcat(prefix,folders{1,i},'/','labels.mat');
        [trainingset,datalabels,data2,lables2] = gettrainData(datanum(1,i),0);
        save(datapath,'trainingset'); 
        save(labelpath,'datalabels');
    end
end
