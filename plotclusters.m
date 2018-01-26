function plotclusters()
    load('agg_sub_plot/complete.mat','agg_activation');
    complete = agg_activation; 
    load('agg_sub_plot/ward.mat','agg_activation');
    ward = agg_activation; 
    load('agg_sub_plot/average.mat','agg_activation');
    average= agg_activation; 
    subplotsim(complete,ward,average);
end

function subplotsim(complete,ward,average)
    fig = figure;   
    subplot(3,1,1);plotone(complete);
    title('Distance Between Clusters: Complete')
    xlabel('Idea');ylabel('Activation')

    subplot(3,1,2);plotone(ward);
    title('Distance Between Clusters: Ward')
    xlabel('Idea');ylabel('Activation')

    subplot(3,1,3);plotone(average);
    title('Distance Between Clusters: Average')
    xlabel('Idea');ylabel('Activation')
end

function plotone(simmatrix)
    colormap('hot');
    imagesc(simmatrix);
    caxis([0.0,1.0]);
end
