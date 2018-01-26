function batchtrainword(hiddenSize,times)
    % data dimention : 2332 * 25
    % data = importdata('word3data/doc_topic_prob_Reweighted.txt'); % new sports data
    data = importdata('word3data_old/doc_topic_prob_NEW_sports_25.txt'); % old sports data
    % new data dimention : 25*2332
    % old data dimention : 25*2583
    data = data';
    d_name = strcat(strcat(num2str(hiddenSize),'_'),num2str(times));
    % dirpath = strcat('word3data/',d_name); % new sports data directory
    dirpath = strcat('word3data_old/',d_name); % new sports data directory
    if ~exist(dirpath)
        mkdir(dirpath);
    else
        rmdir(dirpath,'s'); 
        mkdir(dirpath);
    end
      
    visibleSize = size(data,1);
    data_no = size(data,2);
    Error = zeros(1,data_no);
    SAE_ERR = []; % measure each SAE's average reconstruction error.
    a2 = zeros(hiddenSize,data_no);
    for time = 1:times
        theta_word = trainword3(visibleSize,hiddenSize,data); 
        W1 = reshape(theta_word(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
        W2 = reshape(theta_word(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
        b1 = theta_word(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
        b2 = theta_word(2*hiddenSize*visibleSize+hiddenSize+1:end);
        Z2 = W1*data+repmat(b1,1,data_no); % should be marked when checking
        a_2 = sigmoid(Z2);
        a2 = a2 + a_2;
        Z3 = W2*a_2+repmat(b2,1,data_no); % W2(64*25)
        a_3 = sigmoid(Z3);
        squareError = (a_3-data).^2; %25*2332
        Error = Error+sqrt(sum(squareError)); %reconstruciton error for each SAE
        average_ERR = sum(sqrt(sum(squareError)))/data_no;
        SAE_ERR = [SAE_ERR,average_ERR];

        fex = '.mat';
        w1f = strcat(dirpath,strcat(strcat('/encoder',num2str(time)),fex));
        w2f = strcat(dirpath,strcat(strcat('/decoder',num2str(time)),fex));
        new_weights = norweights(W1);
        save(w1f,'new_weights');
        save(w2f,'W2');
    end    
    Error= Error./repmat(times,1,data_no);
    a2 = a2./repmat(times,hiddenSize,data_no);
    [B,I]= sort(Error);
    word_results = [I',B'];
    filename = strcat(strcat('/wordresults_25_',num2str(hiddenSize)),'hidden.mat');
    save(strcat(dirpath,filename),'word_results');
    [bb,ii] = sort(SAE_ERR);
    disp(ii);

    M = csvread('Sports_SAE/IndexData_10.csv',1,0);  
    C = M(:,1);
    drawsave(a2,I,C,dirpath,hiddenSize);

end

function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end

function[new_weights] = norweights(W1)
    row = size(W1,1);
    col = size(W1,2);
    sumvalue = sum(W1,2);  
    valmap = repmat(sumvalue,1,col);
    new_weights = W1./valmap;
end

function drawsave(a_2,error_index,cluster_index,dirpath,hiddenSize)
    displaysim(a_2(:,error_index));
    filename = strcat(dirpath,strcat(strcat('/sortbyerror_',num2str(hiddenSize)),'hidden.mat'));
    disp(filename)
    sortbyerror = a_2(:,error_index);
    save(filename,'sortbyerror');

    displaysim(a_2(:,cluster_index));
    sortbycluster = a_2(:,cluster_index);
    filename = strcat(dirpath,strcat(strcat('/sortbycluster_',num2str(hiddenSize)),'hidden.mat'));
    save(filename,'sortbycluster');

    Z = linkage(a_2','ward','euclidean');
    dendrogram (Z);

    %A = sortsimilarity(a_2);
    %displaysim(A);
end
