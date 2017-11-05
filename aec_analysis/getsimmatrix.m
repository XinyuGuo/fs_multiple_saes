function [m]=getsimmatrix(features,features2)
% Usage: simaec1 = heatmapnosmooth(featurematix)
    if nargin < 2
        features2 = 0;
    end
    if features2==0
        num_f = size(features,1);
        dist = pdist(features);
        minimum = min(dist(:));
        heatmatrix = triu(ones(num_f),1);
        mt= heatmatrix';
        mt(mt==1) = dist;
        idx = triu(true(size(heatmatrix)), 1)';
        h_transpose = mt';
        h_transpose(idx) = dist;
        %h_transpose;
        m = h_transpose;
    else
        m = pdist2(features,features2);        
    end    
        %heat = HeatMap(h_transpose);
        %fig=figure('Position',[250 250 500 500]);
        %save('similarity.mat','m')
        disp('the smallest similarity value in the nomoothing matrix is:')
        disp(minimum);
        %fig = figure;
        %colormap('hot');
        %imagesc(m);
        %colorbar;
        %caxis([2,7]);
