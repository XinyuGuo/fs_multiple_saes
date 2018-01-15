function trainword2()
    hidden = 100; visible = 1000;

    load('LexicalNetworks/W_Word_Corel_pos.mat','W_Word_Corel_pos');
    W_Word_Corel_pos = W_Word_Corel_pos(1:1000,:);
    theta_Corel_50 = train(1,W_Word_Corel_pos);
    [W1,W2,b1,b2] = get_weights(theta_Corel_50,hidden,visible);
    new_weights = norweights(W1);
    [Corel_50,Corel_50_rec] = shrinkdim(W1,b1,W2,b2,W_Word_Corel_pos);
    disp(size(Corel_50));
    disp(size(Corel_50_rec));
    disp(size(new_weights));
    save('Corel_50.mat','Corel_50');
    save('Corel_50_weights.mat','new_weights')
    save('Corel_50_rec.mat','Corel_50_rec');

    %load('LexicalNetworks/W_Word_Joint.mat','W_Word_Joint');
    %theta_Joint_50 = train(1,W_Word_Joint);
    %[W1,W2,b1,b2] = get_weights(theta_Joint_50,hidden,visible);
    %new_weights = norweights(W1);
    %[Joint_50,Joint_50_rec] = shrinkdim(W1,b1,W2,b2,W_Word_Joint);
    %disp(size(Joint_50));
    %save('Joint_50.mat','Joint_50');
    %save('Joint_50_weights.mat','new_weights')

    %load('LexicalNetworks/W_Word_Log_pos.mat','W_Word_Log_pos');
    %theta_Log_50 = train(1,W_Word_Log_pos);
    %[W1,W2,b1,b2] = get_weights(theta_Log_50,hidden,visible);
    %new_weights = norweights(W1);
    %[Log_50,Log_50_rec] = shrinkdim(W1,b1,W2,b2,W_Word_Log_pos);
    %disp(size(Log_50));
    %save('Log_50.mat','Log_50');
    %save('Log_50_weights.mat','new_weights')
end

function[new_weights] = norweights(W1)
    row = size(W1,1);
    col = size(W1,2);
    sumvalue = sum(W1,2);  
    valmap = repmat(sumvalue,1,col);
    new_weights = W1./valmap;
end

function[W1,W2,b1,b2]= get_weights(theta,hiddenSize,visibleSize)
    W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
end

function [data_small,a_3] = shrinkdim(W1,b1,W2,b2,data)
    number = size(data); % dimension : 64*10000
    m = number(2); % m = 10000
    Z2 = W1*data+repmat(b1,1,m); % should be marked when checking
    a_2 = sigmoid(Z2);
    Z3 = W2*a_2+repmat(b2,1,m); % W2(64*25)
    a_3 = sigmoid(Z3);
    disp(a_2);

    squareError = (a_3-data).^2;
    singleSquareError = sum(squareError);
    %size(singleSquareError)
    %singleSquareError
    [B,I]= sort(singleSquareError);
    %word_results = [I',B'];
    data_small = a_2(:,I);
    %save('wordresults_25.mat','word_results');
end

function sigm = sigmoid(x)
    sigm = 1 ./ (1+exp(-x));
end
