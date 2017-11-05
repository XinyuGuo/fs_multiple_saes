function dispFDistribution()
   load './aeccost_3000/backup_200/selectedhogfeatures.mat' 
   load './aeccost_3000/backup_200/hogfeatures.mat' 
   load './aeccost_3000/backup_200/clusterfeatures.mat'
   load './aeccost_3000/backup_200/evenlyfeatures.mat'

   evenlyfeaturesSim = pdist(evenlyfeatures);
   selectedfeaturesSim = pdist(selectedhfeatures);
   clusterfeaturesSim = pdist(clusterfeatures);
   allfeaturesSim = pdist(hogfeatures);

   esetmin = min(evenlyfeaturesSim)
   esetmax = max(evenlyfeaturesSim)
   esetmean = mean(evenlyfeaturesSim)
   evari = var(evenlyfeaturesSim)

   setmin = min(selectedfeaturesSim);
   setmax = max(selectedfeaturesSim);
   setmean = mean(selectedfeaturesSim);
   svari = var(selectedfeaturesSim);

   csetmin = min(clusterfeaturesSim);
   csetmax = max(clusterfeaturesSim);
   csetmean = mean(clusterfeaturesSim);
   cvari = var(clusterfeaturesSim);

   poolmin = min(allfeaturesSim);
   poolmax = max(allfeaturesSim);
   poolmean = mean(allfeaturesSim);

   emeasuremean = (esetmean - poolmean)/poolmean;
   emeasuremax = 1-((poolmax - esetmax)/poolmax); 
   emeasuremin = (esetmin - poolmin)/(poolmax-poolmin);

   measuremean = (setmean - poolmean)/poolmean;
   measuremax = 1-((poolmax - setmax)/poolmax); 
   measuremin = (setmin - poolmin)/(poolmax-poolmin);
   
   cmeasuremean = (csetmin- poolmean)/poolmean;
   cmeasuremax = 1-((poolmax - csetmax)/poolmax); 
   cmeasuremin = (csetmin - poolmin)/(poolmax-poolmin);

   %disp(measuremean);
   %disp(measuremax);
   %disp(measuremin);
   %
   %
   %disp(cmeasuremean);
   %disp(cmeasuremax);
   %disp(cmeasuremin);

   %disp(emeasuremean);
   %disp(emeasuremax);
   %disp(emeasuremin);

   %figure
   %hist(allfeaturesSim,200),xlim([poolmin poolmax]);
   %figure
   %hist(selectedfeaturesSim,100),xlim([poolmin poolmax]);
   %hist(clusterfeaturesSim,100),xlim([poolmin poolmax]);
   hist(evenlyfeaturesSim,100),xlim([poolmin poolmax]);
   
end
