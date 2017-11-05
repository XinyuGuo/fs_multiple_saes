function combinefeatures()
   prefix = 'aeccost_30000/'; 
   % combine weights from 10 auto-encoders 
   f_filename = 'feature';
   f_tail = '.mat';
   features= [];
   for i = 1:10
       name = strcat(strcat(f_filename,num2str(i)),f_tail);
       filepath = strcat(prefix,name);
       disp(filepath);
       load(filepath,'feature');
       features = [features;feature];
   end
   matfile = strcat(prefix,'featurepack.mat');
   save(matfile,'features');
   disp('combine features!');
