function [covm] = getcovmatrix(feature)
    numf = size(feature,1);
    fid= 1;
    covm = [];
    for i = fid:numf
        index = fid+1;     
        while (index <= numf)
            covm =[covm,cov(feature(fid,:),feature(index,:))];
            index = index +1;
        end
    end
    disp(size(covm));
end
