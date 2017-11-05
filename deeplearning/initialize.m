function theta = initialize(hiddenSize,visibleSize)
%% Initialize parameters randomly based on layer sizes.
%hiddenSize = 200;
%visibleSize = 784;
%if layer ==1 
    load('layer1.mat');
    load('./../aec_analysis/aeccost_10000/features.mat'); 
    %r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
    %disp(size(data));
    W1 = features; 
    %W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

    %b1 = zeros(hiddenSize, 1);
    load('./../aec_analysis/aeccost_10000/opttheta1.mat');
    b11 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    w21 = opttheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize);
    b21 = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);

    load('./../aec_analysis/aeccost_10000/opttheta2.mat');
    b12 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    w22 = opttheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize);
    b22 = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);

    %load('./../aec_analysis/aeccost_30000/opttheta3.mat');
    %b13 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    %w23 = opttheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize);
    %b23 = opttheta(2*hiddenSize*visibleSize+hiddenSize+1:end);

    b1 = (b11+b12 )/2;
    W2 = (w21+w22)/2;
    %disp(size(b1));
    b2 = (b21+b22)/2;

    % Convert weights and bias gradients to the vector form.
    % This step will "unroll" (flatten and concatenate together) all 
    % your parameters into a vector, which can then be used with minFunc. 
    % theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
    theta(1:hiddenSize*visibleSize) = W1(:);
    theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize)=W2(:);
    theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize)=b1(:);
    theta(2*hiddenSize*visibleSize+hiddenSize+1:end)=b2(:);
%else
%    load('layer2.mat');
%end
end
