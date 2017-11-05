function [train,l,train2,l2,test,tl] = gettrainData(t_num,t_num2)
    traindata = loadMNISTImages('mnist/train-images.idx3-ubyte'); %784*60000
    trainlabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
    trainlabels(trainlabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
    %disp(size(traindata));
    %disp(size(trainlabels));
    for k = 1:10
        labels{k} = find(trainlabels==k);
    end
    l = length(labels{1});
    p = randperm(l);
    pl = p(1:t_num);
    pr{1} = p(t_num+1:l);
    datalabels =[trainlabels(pl)];
    train = [traindata(:,pl)];
    %disp(size(train));
    for j = 2:10
        l = length(labels{j});
        p = randperm(l);
        pl = p(1:t_num);
        pr{j} = p(t_num+1:l);
        datalabels = [datalabels;trainlabels(pl)];
        train=[train,traindata(:,pl)];
    end
    %disp(size(datalabels));
    l = length(pr{1});
    p = randperm(l);
    pl = p(1:t_num2);
    datalabels2=[trainlabels(pl)];
    train2= [traindata(:,pl)];
    for j = 2:10
        l = length(pr{j});
        p = randperm(l);
        pl = p(1:t_num2);
        datalabels2 = [datalabels2;trainlabels(pl)];
        train2=[train2,traindata(:,pl)];
    end
    l = datalabels;
    l2 = datalabels2; 
