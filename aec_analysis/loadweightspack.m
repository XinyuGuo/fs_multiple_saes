function[weights] = loadweightspack()
   prefix = 'aeccost_30000/';
   filename = 'weights.mat'; 
   filepath = strcat(prefix,filename);
   load(filepath,'weights');
