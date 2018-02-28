function [a_3]= wordresults()
    %load('data25.mat','data25');
    load('theta_word_25.mat','theta_word');
    data500 = importdata('textdata/DocWord_sort_sports.txt'); 
    %data25 = importdata('textdata/doc_topic_prob_NEW_sports_25.txt'); 
    %data25 = data25';
    data500 = data500';
    data500train = data500(1:500,:);
    %size(data25)
    size(data500train)
    visibleSize = 500;
    hiddenSize = 50;
    W1 = reshape(theta_word(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
    W2 = reshape(theta_word(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
    b1 = theta_word(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
    b2 = theta_word(2*hiddenSize*visibleSize+hiddenSize+1:end);

    %number = size(data25); % dimension : 64*10000
    number = size(data500train);
    m = number(2); % m = 10000
    %Z2 = W1*data25+repmat(b1,1,m); % should be marked when checking
    Z2 = W1*data500train+repmat(b1,1,m); % should be marked when checking
    a_2 = sigmoid(Z2);
    %size(a_2)
    Z3 = W2*a_2+repmat(b2,1,m); % W2(64*25)
    a_3 = sigmoid(Z3);
    %squareError = (a_3-data25).^2;
    squareError = (a_3-data500train).^2;
    singleSquareError = sqrt(sum(squareError));
    disp('here');
    size(singleSquareError)
    Index = (1:2583); 
    sqrerr = [Index;singleSquareError];
    %save('sqrerr.mat','sqrerr');
    save('encoder_docword_500_hidden.mat','W1');
    save('activation_docword_500_hidden.mat','a_3');

    %singleSquareError
    %[B,I]= sort(singleSquareError);
    reconstruction_err = sum(singleSquareError)/m 
    %word_results = [I',B'];
    %save('wordresults_25.mat','word_results');

    %displaysim(a_2(:,I));
    %M = csvread('Sports_SAE/IndexData_10.csv',1,0);  
    %displaysim(a_2(:,M(:,1)));
    %A = sortsimilarity(a_2);
    %displaysim(A);
    %displaysim(W1)
    %displaysim(W2)

    %Z = linkage(a_2','average','euclidean');
    %[O,P,Q] = dendrogram (Z);
    %idea_index = [1:m]';
    %index_cluster = [idea_index,P]
    %save('average_index_cluster.mat','index_cluster');
    %[Bb,Ii]= sort(P);  
    %agg_activation = a_2(:,Ii);
    %save('average.mat','agg_activation');
    %displaysim(a_2(:,Ii));
end

function sigm = sigmoid(x)  
    sigm = 1 ./ (1 + exp(-x));
end

function [A] = sortsimilarity(a2) % 10 * 2583
   index = randi(size(a2,2)); 
   activation = a2(:,index);
   A = [activation];
   a2(:,index) = [];
   while size(a2,2)~=0
       similarity = [];
       for i = 1 : size(a2,2)
           cosine = pdist([activation,a2(:,i)]','euclidean');
           similarity = [similarity,cosine];
       end
       [B,I] = sort(similarity,'descend');
       activation = a2(:,I(1));
       A = [A,activation];
       a2(:,I(1)) = [];
   end
end
