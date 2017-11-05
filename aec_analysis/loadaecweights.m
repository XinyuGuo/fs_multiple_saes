function [w] = loadaecweights(id)
    prefix = 'aeccost_30000/';
    filename = 'aec';
    aecid = num2str(id);
    filepath= strcat(strcat(filename,aecid),'.mat');
    filepath= strcat(prefix,filepath);
    load(filepath,'w');
