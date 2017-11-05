function theta = initializeParameters(hiddenSize, visibleSize,layer)
if layer == 1
    r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
    load('./../aec_analysis/aeccost_30000/newfeatures.mat'); 
%disp(size(data));
    W1 = data; 
    W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

    b1 = zeros(hiddenSize, 1);
    load('./../aec_analysis/aeccost_30000/opttheta1.mat');
    b11 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    load('./../aec_analysis/aeccost_30000/opttheta2.mat');
    b12 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    load('./../aec_analysis/aeccost_30000/opttheta3.mat');
    b13 = opttheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    b1 = (b11 + b12 + b13)/3;
    %disp(size(b1));
    b2 = zeros(visibleSize, 1);
    % Convert weights and bias gradients to the vector form.
    % This step will "unroll" (flatten and concatenate together) all 
    % your parameters into a vector, which can then be used with minFunc. 
    theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
elseif layer ==2
%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end
