Baseline = [0.6000 0.7600 0.8400 0.6600 0.7400 0.7400 0.7400 0.8200 0.7800 0.7400];
Enhance =  [0.7200 0.8000 0.9000 0.6400 0.7800 0.7400 0.7200 0.8400 0.8000 0.7600];
figure;
boxplot([Baseline',Enhance'],'whisker',5,'notch','off','labels',{'Baseline','Library Enhancing'});
%title('Miles per Gallon by Vehicle Origin')
xlabel('Deep Learning Algorithms')
ylabel('CLassification Accuracy')
