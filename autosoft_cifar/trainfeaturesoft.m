function [stackedAETheta] = trainfeaturesoft(hsize,vsize,numofClasses,traindata,trainlabels,testdata,testlabels,paras)
    saeSoftmaxTheta = paras{1,2};
    load('./../aec_analysis/aeccost_3000/features.mat'); 
    load('./../aec_analysis/aeccost_3000/b1.mat'); 
    bias = b1;
    softtraindata = getHighFeatures(features,bias,hsize,vsize,traindata); 

    lambda = 3e-3;         % weight decay parameter       
    stack = cell(1,1);
    stack{1}.w = features;
    stack{1}.b = b1; 
    [stackparams, netconfig] = stack2params(stack);

    options.maxIter = 100;
    disp('train softmax');
    softmaxModel = softmaxTrain(hsize, numofClasses, lambda, ...
                            softtraindata, trainlabels, options);
    saeSoftmaxOptTheta = softmaxModel.optTheta(:);
    stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

    %[pred] = autoSoftPredict(stackedAETheta, vsize, hsize, ...
    %                      numofClasses, netconfig, testdata);
    %beforeTunepred = mean(testlabels(:) == pred(:));
    %fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', beforeTunepred * 100);
end
