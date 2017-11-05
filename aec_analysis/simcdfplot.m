function simcdfplot()
    load('./aeccost_3000/backup_40_similarity/hogfeatures.mat','hogfeatures');   
    similarity40 = checkSimilarFeatures(hogfeatures,40); 
    h = cdfplot(similarity40);
    set(h,'LineWidth',2);
    hold on
    load('./aeccost_3000/backup_200_similarity/hogfeatures.mat','hogfeatures');   
    similarity200 = checkSimilarFeatures(hogfeatures,200); 
    h = cdfplot(similarity200);
    hold off
    set(h,'color','r','LineWidth',2);
end
