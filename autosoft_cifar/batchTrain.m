function batchTrain()
    %dataNum = [1200,1400,1600,1800];
    dataNum = [1000];

    theta1 = initializeParameters(200,784); 
    theta2 = initializeParameters(200,200); 
    saeSoftmaxTheta = 0.005*randn(200*10,1);
    paras = cell(1,3);
    paras{1,1} = theta1;
    paras{1,2} = theta2;
    paras{1,3} = saeSoftmaxTheta;

    fileID = fopen('result.txt','w');
    for i= 1:size(dataNum,2) 
       num = dataNum(1,i);
       fprintf(fileID,'data number:%d\n',10*num);
       deepTrain(num,fileID,paras); 
    end
    fclose(fileID);
