function trainword()
   load('data25.mat','data25'); 
   %load('traindata.mat','trainingset'); 
   theta_word = train(1,data25);
   save('theta_word_25.mat','theta_word');
end
