function batchtrainword(hiddenSize,times)
    % data dimention : 2332 * 25
    data = importdata('word3data/doc_topic_prob_Reweighted.txt');
    % data dimention : 25*2332
    data = data';
    d_name = strcat(strcat(num2str(hiddenSize),'_'),num2str(times));
    dirpath = strcat('word3data/',d_name);
    if ~exist(dirpath)
        mkdir(dirpath);
    else
        rmdir(dirpath,'s'); 
        mkdir(dirpath);
    end
      
    visibleSize = size(data,1);
    data_no = size(data,2);
    Error = zeros(1,data_no);
    SAE_ERR = [] % measure each SAE's average reconstruction error.
    for time = 1:times
        theta_word = trainword3(visibleSize,hiddenSize,data); 
        W1 = reshape(theta_word(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
        W2 = reshape(theta_word(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
        b1 = theta_word(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
        b2 = theta_word(2*hiddenSize*visibleSize+hiddenSize+1:end);
        Z2 = W1*data+repmat(b1,1,data_no); % should be marked when checking
        a_2 = sigmoid(Z2);
        Z3 = W2*a_2+repmat(b2,1,data_no); % W2(64*25)
        a_3 = sigmoid(Z3);
        squareError = (a_3-data).^2; %25*2332
        Error = Error+sum(squareError); %reconstruciton error for each SAE
        average_ERR = sum(sum(squareError))/data_no;
        SAE_ERR = [SAE_ERR,average_ERR];

        fex = '.mat';
        w1f = strcat(dirpath,strcat(strcat('/encoder',num2str(time)),fex));
        w2f = strcat(dirpath,strcat(strcat('/decoder',num2str(time)),fex));
        save(w1f,'W1');
        save(w2f,'W2');
    end    
    Error= Error./repmat(times,1,data_no);
    [B,I]= sort(Error);
    word_results = [I',B'];
    save(strcat(dirpath,'/wordresults_25.mat'),'word_results');
    [bb,ii] = sort(SAE_ERR);
    disp(ii);
end

function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end
