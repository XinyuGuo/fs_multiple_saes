function displaycorr(corr,index)
    %heat = HeatMap(simmatrix);
    %fig=figure('Position',[250 250 500 500]);
    %save('similarity.mat','m')
    %disp('the smallest similarity value is:')
    %disp(minimum);
    if nargin < 2
        index = 0; 
    end

    u = triu(corr,1);
    maxvalue = max(max(u));
    minvalue = min(min(corr));
    fig = figure;
    %mymap = [0 0 0
    %        1 0 0
    %        0 1 0
    %        0 0 1
    %        1 1 1];
    %colormap(mymap);
    colormap('hot');
    if index ==0
        imagesc(corr);
    else
        corr(:,index);
        imagesc(corr);
    end

    colorbar;
    caxis([minvalue,maxvalue]);
    disp(minvalue);
    disp(maxvalue);
