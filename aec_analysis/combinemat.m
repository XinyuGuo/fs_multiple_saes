function combinemat()
   clear all;
   prefix = 'aeccost_30000/'; 
   % combine weights from 10 auto-encoders 
   w_filename = 'aec';
   w_tail = '.mat';
   weights= [];
   for i = 1:10
       name = strcat(strcat(w_filename,num2str(i)),w_tail);
       filepath = strcat(prefix,name);
       load(filepath,'w');
       weights = [weights;w];
   end
   matfile = strcat(prefix,'weights.mat');
   save(matfile,'weights');
   disp('combine weights!');


