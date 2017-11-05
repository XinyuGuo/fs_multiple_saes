% cost-training epoches
function displayCost()
    load('aeccost_30000/costvalue.mat','costvalue');

    iteration = 1:400;
    h = figure;
    
    plot(iteration,costvalue{10},'LineWidth',2);
end
