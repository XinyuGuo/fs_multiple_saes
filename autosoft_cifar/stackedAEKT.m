function [beforeTunepred,afterTunepred] = stackedAEKT (sae1OptTheta,sae2OptTheta,traindata,trainlabels,testdata,testlabels,knowledgeTransfer,borrow,fid,paras)
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

% disp('datasize');
% disp(size(traindata));
% pause;
inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
% trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
% trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');
% 
% trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================

%%======================================================================
%
% set up application flags
% numbers= [5,6,7,8,9];
% samplenumber = 10; 
% fullset = false;
DEBUG  = true;
Show_Weights = false;
Results_To_File = false;

% construct training dataset 
%[traindata,trainlabels]=constructDataset(trainData,trainLabels,samplenumber,numbers,fullset);

%tic;

%% STEP 2: Train  the first sparse autoencoder
if knowledgeTransfer
    sae1Theta = sae1OptTheta;
else
    if borrow
        sae1OptTheta = initialize(hiddenSizeL1, inputSize);
    else
        sae1Theta = paras{1,1};
    end
end;
% disp('theta size');
% disp(size(sae1OptTheta));
% pause;

%[cost, grad] = sparseAutoencoderCost(sae1Theta, inputSize, hiddenSizeL1, lambda, ...
%                                    sparsityParam, beta, traindata);
                                 
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 60;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
options.Corr = 10;

if ~borrow
    disp('train first layer');
    [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                    inputSize, hiddenSizeL1, ...
                                    lambda, sparsityParam, ...
                                    beta, traindata), ...
                                    sae1Theta, options);
end
% -------------------------------------------------------------------------

% Visualize weights
if Show_Weights
W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
[h,array1]=display_network(W1');
end;
%save 'sae1OptTheta.mat' sae1OptTheta;
%%======================================================================
%% STEP 3: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
if knowledgeTransfer
    sae2Theta = sae2OptTheta;
else
    sae2Theta = paras{1,2};
    %sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
    %if borrow
    %    sae2Theta = initialize(hiddenSizeL2, hiddenSizeL1 ,2);
    %else
    %    sae2Theta = initializeParameters(hiddenSizeL2,hiddenSizeL1,2);
    %end
end

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                            inputSize, traindata);

[cost, grad] = sparseAutoencoderCost(sae2Theta, hiddenSizeL1, hiddenSizeL2, lambda, ...
                                     sparsityParam, beta, sae1Features);
                                 
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 60;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

disp('train second layer');
[sae2OptTheta, cost] = minFunc(@(p) sparseAutoencoderCost(p, ...
                               hiddenSizeL1, hiddenSizeL2, ...
                               lambda, sparsityParam, ...
                               beta, sae1Features), ...
                               sae2Theta, options);
%%======================================================================
% Visualize weights
if Show_Weights
W2 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
[h,array2]=display_network(W2');
end;
%save 'sae2OptTheta.mat' sae2OptTheta;
%%======================================================================
%% STEP 4: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
%  saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
saeSoftmaxTheta = paras{1,3};
%if borrow
%    %sae2Theta = initialize(hiddenSizeL2, hiddenSizeL1 ,2);
%    load('softmaxTheta.mat');
%else
%    saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
%    %sae2Theta = initializeParameters(hiddenSizeL2,hiddenSizeL1,2);
%    save('softmaxTheta.mat','saeSoftmaxTheta');
%end
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

options.maxIter = 10;
disp('train softmax');
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainlabels, options);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% -------------------------------------------------------------------------


%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%
if ~borrow
    lambda = 1e-4; 
    addpath minFunc/ 
    options.Method = 'lbfgs';  
    options.maxIter = 5;	  
    options.display = 'on'; 
    disp('fine tune');
    [stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, hiddenSizeL2, ... 
                                     numClasses, netconfig, lambda, ... 
                                     traindata, trainlabels), stackedAETheta, options); 
end
% -------------------------------------------------------------------------

%%======================================================================

%toc;

%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
% testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
% testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');
% 
% testLabels(testLabels == 0) = 10; % Remap 0 to 10
% 
% if DEBUG
%    test_number = round(size(dataIndex,1)/5);
%    testIndex = randi(length(testLabels),test_number,1);
%    testdata = testData(:,testIndex);
%    testlabels = testLabels(testIndex);
% else
%    testdata = testData;
%    testlabels = testLabels;
% end

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testdata);

beforeTunepred = mean(testlabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', beforeTunepred * 100);

if ~borrow
    [pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                              numClasses, netconfig, testdata);
    afterTunepred = mean(testlabels(:) == pred(:));
    fprintf('After Finetuning Test Accuracy: %0.3f%%\n', afterTunepred * 100);
end

%fprintf(fid,'Before Finetuning Test Accuracy: %0.3f%%\n', beforeTunepred * 100);
%s = ' ';
%fprintf(fid,'%s',s);
%fprintf(fid,'After Finetuning Accuracy: %0.3f%%\n', afterTunepred * 100);
%

if Results_To_File
fid=fopen('result.txt','w');
fprintf(fid,'%f',beforeTunepred);
fprintf(fid,'%f',afterTunepred);
s = ' ';
fprintf(fid,'%s',s);
fclose(fid);
end;

end
