function [similarity]= checkSimilarFeatures(hogfeatures,hiddensize)
   %load './aeccost_3000/backup_200_similarity/hogfeatures.mat'; 
   size(hogfeatures)
   hidden = hiddensize;
   similarity = [];
   for i = 1:10 
     thisSAEfeatures = hogfeatures(hidden*(i-1)+1:hidden*i,:); 
     anotherSAEfeatures = hogfeatures; 
     anotherSAEfeatures(hidden*(i-1)+1:hidden*i,:)=[]; 
     for j = 1:9
        thatSAEfeatures = anotherSAEfeatures(hidden*(j-1)+1:hidden*j,:); 
        D = pdist2(thisSAEfeatures,thatSAEfeatures);
        sim = sort(D,2);
        similarity = [similarity;sim(:,1)];
     end
   end
   %cdfplot(similarity)
   size(similarity)
   %hist(similarity,100),xlim([0 1.5]);
   %set(gca,'view',[180,-180]);
   mean(similarity)
   max(similarity)
   min(similarity)
   var(similarity)
end
