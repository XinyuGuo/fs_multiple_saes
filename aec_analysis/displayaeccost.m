% Usage : displayaeccost('costvalue.mat') 
% plot iteratios vs cost for 10 different autoencoders
function displayaeccost(data)
    disp(data);
    data = strcat('aeccost_30000/',data);
    if exist(data,'file')
        disp('using .mat file!');
        load(data,'costvalue');
    else
        filename = 'aeccost_30000/aeccost.xlsx';
        range = 'D:D';
        costvalue = cell(10,1);
        for i = 1:10
            sheet = strcat('Sheet',num2str(i));
            disp(sheet);
            costvalue{i} = xlsread(filename,sheet,range);
        end
        save('aeccost_30000/costvalue.mat','costvalue');
    end
   
    iteration = 1:400;
    h = figure;
    for i = 1:10
        subplot(2,5,i), plot(iteration,costvalue{i},'LineWidth',2);
        set(gca,'fontsize',8);
        %plot(iteration,costvalue{i});
        %xlabel('epochs')
        %ylabel('cost')
        t = 'SAE';
        id = num2str(i);
        thistile = strcat(t,id);
        title(thistile,'FontSize',10);

        %set(h,'Units','Inches');
        %pos = get(h,'Position');
        %set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        %print(h,'filename','-dpdf','-r0')
    end
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(h,'filename','-dpdf','-r0')
