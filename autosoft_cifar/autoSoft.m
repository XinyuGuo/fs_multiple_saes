function autoSoft(traindata,trainlabels,testdata,testlabels,borrow,paras,anum,snum,i)
inputSize = 3072;
hiddenSize = 400;
numClasses =10;
disp(size(traindata));
if borrow
    saeSoftmaxTheta = paras{1,2};
    %load('./../aec_analysis/aeccost_3000/features.mat'); 
    
    %load('./../aec_analysis/aeccost_3000/b1.mat'); 
    load('./../aec_analysis/aeccost_3000/backup_200/lassofeatures.mat'); 
   
    load('./../aec_analysis/aeccost_3000/backup_200/lassobias.mat'); 
    %bias = b1;
    bias=lassobias;
    b1=lassobias;
    features = lassofeatures;
    softtraindata = getHighFeatures(features,bias,hiddenSize,inputSize,traindata); 
    %load('../aec_analysis/aeccost_10000/activation.mat');
    %softtraindata = activation; 
    %disp('******************888888');
    %disp(size(activation));
    lambda = 3e-3;         % weight decay parameter       
    %options.maxIter = 100;
    %disp('train softmax');
    %softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
    %                            sae2Features, trainlabels, options);
    %saeSoftmaxOptTheta = softmaxModel.optTheta(:);
    stack = cell(1,1);
    stack{1}.w = features;
    stack{1}.b = b1; 

    [stackparams, netconfig] = stack2params(stack);
else
    theta = paras{1,1};
    saeSoftmaxTheta = paras{1,2};

    sparsityParam = 0.1;   % desired average activation of the hidden units.
    lambda = 3e-3;         % weight decay parameter       
    beta = 3;              % weight of sparsity penalty term       

    [cost, grad] = sparseAutoencoderCost(theta, inputSize, hiddenSize, lambda, ...
                                         sparsityParam, beta, traindata);

    addpath minFunc/
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                              % function. Generally, for minFunc to work, you
                              % need a function pointer with two outputs: the
                              % function value and the gradient. In our problem,
    options.maxIter = anum;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';
    options.Corr = 10;

    disp('train the autoencoder');
    [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                     inputSize, hiddenSize, ...
                                     lambda, sparsityParam, ...
                                     beta, traindata), ...
                                     theta, options);
    %save('../aec_analysis/aeccost_10000/opt.mat','sae1OptTheta');
    softtraindata = feedForwardAutoencoder(sae1OptTheta,hiddenSize,inputSize,traindata); 
    stack = cell(1,1);
    stack{1}.w = reshape(sae1OptTheta(1:hiddenSize*inputSize), ...
                         hiddenSize, inputSize);
    stack{1}.b = sae1OptTheta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);

    [stackparams, netconfig] = stack2params(stack);
     
end

options.maxIter = snum;
disp('train softmax');
softmaxModel = softmaxTrain(hiddenSize, numClasses, lambda, ...
                            softtraindata, trainlabels, options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

parapath = './cifar-10-batches-mat/';
filename = strcat('stackedParas',num2str(i),'.mat'); 
filepath = strcat(parapath,filename);
save(filepath,'stackedAETheta');

[pred] = autoSoftPredict(stackedAETheta, inputSize, hiddenSize, ...
                          numClasses, netconfig, testdata);
beforeTunepred = mean(testlabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', beforeTunepred * 100);
end
