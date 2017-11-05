% some simple , useful matlab instructions used in my experiments.

% Document all smallest value in each similarity matirx. 
% 1 3.9380
% 2 3.6460
% 3 3.9688
% 4 3.9629
% 5 3.8492
% 6 3.5232
% 7 3.6575
% 8 3.9369
% 9 3.9168
% 10 3.6025
% smallest in all: 3.5232

% get similarity matrix after smoothing weights 
% the smallest value in combined similarity matrix is 0.8054
% the largest value in combined similarity matrix is 8.1492
[array,smoothweights] = getsmoothweights(weights');
smoothsim = getsimmatrix(smoothweights);
displaysim(smoothsim);

% sort 2000*2000 similarity matirx.
% simpack.mat contains 2000*2000 similarities. 
% the smallest value in the simpack.mat is 2.2143, the larges value 
% in the simpack.mat is 6.6909
[Y,I] = sort(sum(smoothsim));
new_smoothsim= smoothsim(:,I);

% generat heatmap for the feature matrix 
load feature1.mat
corr =  corrcoef(feature');
