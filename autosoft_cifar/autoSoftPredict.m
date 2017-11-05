function [pred] = autoSoftPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
    softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);
    stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);
    
    z2 = bsxfun(@plus,stack{1}.w*data,stack{1}.b);
    a2 = sigmoid(z2);
    %z3 = bsxfun(@plus,stack{2}.w*a2,stack{2}.b);
    %a3 = sigmoid(z3);
    a3 = exp(softmaxTheta*a2);
    a3 = bsxfun(@rdivide,a3,sum(a3));

    [p,pred] = max(a3,[],1);
end
% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
