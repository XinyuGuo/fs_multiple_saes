function displaysim(simmatrix)
    %heat = HeatMap(simmatrix);
    %fig=figure('Position',[250 250 500 500]);
    %save('similarity.mat','m')
    %disp('the smallest similarity value is:')
    %disp(minimum);
    fig = figure;
    %mymap = [0 0 0
    %        1 0 0
    %        0 1 0
    %        0 0 1
    %        1 1 1];
    %colormap(mymap);
    colormap('hot');
    imagesc(simmatrix);
    colorbar;
    caxis([0.5,8.5]);
