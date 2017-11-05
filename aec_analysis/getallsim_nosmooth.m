function getallsim_nosmooth()
   % caculate all similarity matrix for all of auto-encoders     
   prefix = 'aeccost_30000/';
   fileprefix = 'aecsim_nosmooth_';
   filetail = '.mat';
   for i =1:10
       id = num2str(i);
       filename = strcat(strcat(fileprefix,id),filetail);  
       filepath= strcat(prefix,filename);
       ww = loadaecweights(i); 
       ws = getsimmatrix(ww);
       save (filepath,'ws');
   end
   disp('get all similarity matrix from weights of 10 auto-encoders!');
