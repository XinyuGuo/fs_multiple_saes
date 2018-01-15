function trainword()
   data25 = importdata('textdata/doc_topic_prob_NEW_sports_25.txt'); 
   data25 = data25';
   %load('data25.mat','data25'); 
   %load('traindata.mat','trainingset'); 
   theta_word = train(1,data25);
   save('theta_word_25.mat','theta_word');
   a_3 = wordresults();
   %a_3
   max(max(a_3))
   displaysim(a_3');
   displaysim(data25');
end
