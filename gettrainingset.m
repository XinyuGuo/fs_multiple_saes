function[training] = gettrainingset(num,dataset,labels)
    datalabels = cell(10,1);
    labels(labels==0)=10;
    for digit = 1:10
        position = find(labels==digit);
        datalabels{digit}= position(1:num);     
    end
    l = datalabels{1};
    for digit = 2:10
        l = [l,datalabels{digit}];
    end
    training = dataset(:,l);
    
