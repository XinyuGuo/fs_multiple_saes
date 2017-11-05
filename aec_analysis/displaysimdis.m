function displaysimdis(simmat)
%Usage: plotsimdis(sim) sim: similarity matrix - square matrix
   row = size(simmat,1);
   col = size(simmat,2);
   %border1 = 3;
   %border2 = 4.5;
   %border3 = 6;
   %border4 = 7;
   %bordergroup = cell(4,1);  
   %bordergroup{1}= border1;
   %bordergroup{2}= border2;
   %bordergroup{3}= border3;
   %bordergroup{4}= border4;
% ten bins
   %border1 = 2;
   %border2 = 2.5;
   %border3 = 3;
   %border4 = 3.5;
   %border5 = 4;
   %border6 = 4.5;
   %border7 = 5;
   %border8 = 5.5;
   %border9 = 6;
   %border10 = 6.5;
   %border11 = 7;


   %bordergroup = cell(11,1);  
   %bordergroup{1}= border1;
   %bordergroup{2}= border2;
   %bordergroup{3}= border3;
   %bordergroup{4}= border4;
   %bordergroup{5}= border5;
   %bordergroup{6}= border6;
   %bordergroup{7}= border7;
   %bordergroup{8}= border8;
   %bordergroup{9}= border9;
   %bordergroup{10}= border10;

   border1 = 2;
   border2 = 2.25;
   border3 = 2.5;
   border4 = 2.75;
   border5 = 3;
   border6 = 3.25;
   border7 = 3.5;
   border8 = 3.75;
   border9 = 4;
   border10 = 4.25;
   border11 = 4.5;
   border12 = 4.75;
   border13 = 5;
   border14 = 5.25;
   border15 = 5.5;
   border16 = 5.75;
   border17 = 6;
   border18 = 6.25;
   border19 = 6.5;
   border20 = 6.75;
   %border21 = 7;

   bordergroup = cell(20,1);  
   bordergroup{1}= border1;
   bordergroup{2}= border2;
   bordergroup{3}= border3;
   bordergroup{4}= border4;
   bordergroup{5}= border5;
   bordergroup{6}= border6;
   bordergroup{7}= border7;
   bordergroup{8}= border8;
   bordergroup{9}= border9;
   bordergroup{10}= border10;
   bordergroup{11}= border11
   bordergroup{12}= border12;
   bordergroup{13}= border13;
   bordergroup{14}= border14;
   bordergroup{15}= border15;
   bordergroup{16}= border16;
   bordergroup{17}= border17;
   bordergroup{18}= border18;
   bordergroup{19}= border19;
   bordergroup{20}= border20;
   %bordergroup{21}= border21
   % get truth tables for all categories
   t = gettruthtable(bordergroup,simmat);  

   tablesize = size(t,1);
   yaxis = zeros(tablesize,1);
   xlabel = cell(1,tablesize); 
   for i = 1:tablesize
       curtable = [t{i}];
       yaxis(i,1) = sum(curtable(:));
       s = cell(1,3); 
       if i == 1
            s{1} = 0;s{2}='-';s{3}=bordergroup{1};
            str = sprintf('%0.2f%s',s{:});
       else
            s = {bordergroup{i-1},'-',bordergroup{i}};
            str = sprintf('%0.2f%s',s{:});
       end
       xlabel{i} = str;
       disp(str);
       %fprintf('%d\n',sum(curtable(:)));
   end

   fprintf('%d\n',yaxis);
   fprintf('\n');
   yaxis(1) = yaxis(1) -200;
   yaxis = yaxis/2;
   fprintf('%d\n',yaxis);
   fprintf('\n');

   %xlabel= {'0-3.0','3.0-4.5','4.5-6','6-7'};
   plotdatadistribution(yaxis,xlabel);
   %xbins = [border1 border2 border3 border4];
   %hist(yaxis,xbins);
   %table2 = [t{2}];
   %disp(sum(table2(:)));
   triangle = triu([t{1}],1);
   fprintf('%d',sum(triangle(:)));
   fprintf('\n');
   %size(r)
   %size(c)
   %groups = groupnodes(triangle);
   %printgroups(groups);

function [table]=gettruthtable(group,mat) 
    num = size(group,1);
    dim = size(mat,1);
    table = cell(num,1);

    for index = 1:num
        truthtable= repmat(group{index},dim,dim);
        ta = mat<=truthtable;
        %if index ==4 
        %    disp(sum(ta(:)));
        %    error();
        %end
        if index>1 
            for k = 1:index-1
                ta = ta - [table{k}];
            end
            %disp(sum(truthtable(:)));
        end 
        table{index}=ta;
        %if index == 4 
        %    disp(sum(ta(:)));
        %    error();
        %end
    end

function plotdatadistribution(y,category)
   catnum = size(category,2);
   name = cell(1,catnum);
   for i = 1:catnum
      name{i} = category{i};
   end
   figure;
   % name = {'0-3.0','3.0-4.5','4.5-6','6-7'};
   bar(y);
   x = 1:catnum;
   for i=1:catnum
       th(i) = text(x(i),y(i),num2str(y(i)));
   end
   set(gca,'xticklabel',name);
   set(th,'Horizontalalignment','center','verticalalignment','bottom') ;

%function [nodesgroup] = groupnodes(trium)
%    disp(trium(2,50));
%    nodenum = size(trium,2);
%    nodesgroup = java.util.Stack(); 
%    for nodeid= 1:2
%       [r,c] = find(trium);
%       pos = find(r==nodeid);%find nodes are very similary to the current node.
%       if ~isempty(pos)% exist nodes are very similar to the current node.
%           startnode = java.util.Stack(); 
%           group = java.util.Stack();
%           group.push(nodeid);
%           pos_nodes = c(pos);
%           trium(nodeid,pos_nodes)=0;%get rid of nodes
%           %disp(trium(1,146));
%           for i = 1:size(pos_nodes,1) %push push similar nodes into the stack.
%               startnode.push(pos_nodes(i));
%           end
%           while(~startnode.empty())
%               topnode = startnode.pop();
%               disp(topnode);
%               group.push(topnode);
%               [curr,curc] = find(trium);
%               this_pos = find(curr==topnode);
%               if~isempty(this_pos)
%                   disp('i am here!');
%                   thisnode_pos = curc(this_pos);
%                   disp(thisnode_pos);
%                   disp(trium(50,89));
%                   disp(trium(50,144));
%                   disp(trium(50,175));
%                   trium(topnode,thisnode_pos)=0;
%                   for j = 1:size(thisnode_pos,1)
%                       startnode.push(thisnode_pos(j));
%                   end
%               end
%           end
%           nodesgroup.push(group);
%       end      
%    end
%
%function printgroups(s)
%    groupnum = 0;
%    featurenum =0;
%    while(~s.empty())
%        thisgroup = s.pop(); 
%        featurenum = featurenum + thisgroup.size();
%        groupnum = groupnum+1;
%        fprintf('group : %d \n',groupnum);
%        while(~thisgroup.empty())
%            fprintf('%d ',thisgroup.pop());
%            fprintf('\n');
%        end
%    end
%    fprintf('%d',featurenum);
