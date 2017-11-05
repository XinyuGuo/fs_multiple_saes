function [ws]= loadweightsim(id)
% load one similarity matirx of one auto-encoder to the workspace
    prefix = 'aeccost_30000/';
    fileprefix = 'aecsim'; 
    fileid = num2str(id);
    filetail = '.mat';    
    filename= strcat(strcat(fileprefix,fileid),filetail);
    filepath = strcat(prefix,filename); 
    load(filepath,'ws');
