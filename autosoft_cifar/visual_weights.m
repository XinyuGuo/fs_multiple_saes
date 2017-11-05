function visual_weights()
   load('cifar-10-batches-mat/stackedParas1.mat','stackedAETheta');
   hidden = 200;
   visible = 3072;
   numofClass = 10;
   patches = reshape(stackedAETheta(numofClass*hidden+1:numofClass*hidden+hidden*visible),hidden,32,32,3);
   padding = 1; 
   numofpatches  = ceil(sqrt(size(patches,1))); 
   grid_height = numofpatches * 32 + padding * (numofpatches-1);
   grid_width  = numofpatches * 32 + padding * (numofpatches-1);
   grid = zeros(grid_height,grid_width,3);  

   next_idx = 1;
   y0 = 1; y1 = 32; 
   for i = 1:numofpatches
     x0 = 1; x1 = 32; 
     for j = 1:numofpatches
       if next_idx <= hidden 
         patch = patches(next_idx,:,:,:); 
         low =  min(min(min(patch)));
         high = max(max(max(patch)));
         image = 255.0*(patch-low)/(high-low); 
         grid(y0:y1,x0:x1,:) = image;
	 next_idx = next_idx+1;
       end 
       x0 = x0 + 32 + padding;
       x1 = x1 + 32 + padding;
     end 
     y0 = y0 + 32 + padding;
     y1 = y1 + 32 + padding; 
   end
   imshow(uint8(grid));
   %size(patches)
   %onepatch = patches(166,:);
   %size(onepatch)
   %onepatch = reshape(onepatch,32,32,3);  
   %low  = min(min(min(onepatch)));
   %high = max(max(max(onepatch)));
   %image =255.0* (onepatch-low)/(high-low);
   %imshow(uint8(image));
end
