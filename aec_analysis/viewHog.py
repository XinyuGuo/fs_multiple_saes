import matplotlib.pyplot as plt
import scipy.io as importer
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np

path = './aeccost_10000/features.mat'
mat = importer.loadmat(path)
data = mat['features']
image_1 = np.reshape(data[2,:],[28,28],order='F')
#print data.shape
#print image_1[0,:]
#print image_1[0,:]
#print data[2,:]
#print image_1[0,:]
image_2 = np.reshape(data[5,:],[28,28],order='F')
#print image.shape
#image = color.rgb2gray(data.astronaut())

fd, hog_image= hog(image_1, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=True)
fd2,hog2_image = hog(image_2, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(1, 1), visualise=True)

edist = np.linalg.norm(fd2-fd)
print edist

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image_1, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
