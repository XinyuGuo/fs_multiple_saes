function wordresults()
    load('data25.mat','data25');
    load('theta_word_25.mat','theta_word');
    visibleSize = 25;
    hiddenSize = 125;
    W1 = reshape(theta_word(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    W2 = reshape(theta_word(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    b1 = theta_word(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    b2 = theta_word(2*hiddenSize*visibleSize+hiddenSize+1:end);

    number = size(data25); % dimension : 64*10000
    m = number(2); % m = 10000
    Z2 = W1*data25+repmat(b1,1,m); % should be marked when checking
    a_2 = sigmoid(Z2);
    Z3 = W2*a_2+repmat(b2,1,m); % W2(64*25)
    a_3 = sigmoid(Z3);
    squareError = (a_3-data25).^2;
    singleSquareError = sum(squareError);
    size(singleSquareError)
    %singleSquareError
    [B,I]= sort(singleSquareError);
    word_results = [I',B'];
    save('wordresults_25.mat','word_results');
end

function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end
