function [w] = getweightsfromopt()
    numofopts = 10;
    prefix = 'aeccost_30000/';
    filename = 'opttheta';
    desfilename = 'aec';
    fileextension = '.mat'; 
    fpath = strcat(prefix,filename);
    desfpath = strcat(prefix,desfilename);
    hiddenSize = 200;
    visibleSize = 784;
    for i = 1 : numofopts
        fileid = num2str(i);
        filepath = strcat(fpath,fileid);
        desfilepath = strcat(desfpath,fileid);
        path = strcat(filepath,fileextension);
        despath = strcat(desfilepath,fileextension);
        load(path); 
        w = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
        save(despath,'w');
    end
