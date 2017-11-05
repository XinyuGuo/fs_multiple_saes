function [activation] = getHighFeatures(W1,b1, hiddenSize, visibleSize, data)
    N = size(data);
    sample_N = N(2);
    disp(sample_N);
    disp(size(W1));
    disp(size(data));
    Z2 = W1*data + repmat(b1,1,sample_N);%+ B1;
    activation = sigmoid(Z2);
end
% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
