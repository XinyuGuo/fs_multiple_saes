function batchtrainword(hidden,times)
    data = importdata('word3data/doc_topic_prob_Reweighted.txt');
    d_name = strcat(strcat(num2str(hidden),'_'),num2str(times));
    dirpath = strcat('word3data/',d_name);
    if ~exist(dirpath)
        mkdir(dirpath);
    else
        rmdir(dirpath);   
        mkdir(dirpath);
    end

    for time = 1:times
       theta_word = trainword3(visible,hidden); 
        
    %theta_word = tra(1,data25);
    %save('theta_word_25.mat','theta_word');
end
