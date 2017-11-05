function test()
   parapath = '../aec_analysis/aeccost_3000/';
   filename = strcat('stackedParas',num2str(1),'.mat'); 
   filepath = strcat(parapath,filename); 
   stackedAETheta =[1,2,3]; 
   save(filepath,'stackedAETheta'); 
end
