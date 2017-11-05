function pickFeaturesEvenly()
   load './aeccost_3000/hogfeatures.mat'; 
   l2norm = diag(sqrt(hogfeatures*hogfeatures')); 
   [A,I] = sort(l2norm);
   indices = I(1:200);
   evenlyfeatures = hogfeatures(indices,:); 
   size(evenlyfeatures)
   save('./aeccost_3000/evenlyfeatures.mat','evenlyfeatures');
end
