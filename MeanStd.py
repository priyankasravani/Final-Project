from PIL import Image 
import os
import numpy as np
listimgs = []
dir = "locate"#Location of the folder class in which all the images
# Loading images in numpy-arrays
for imgclass in os.listdir(dir):
    for fname in os.listdir(dir + imgclass):
        img = Image.open(dir + imgclass + '/' + fname)
        # img_arr = np.array(img) / 255.
        listimgs.append(img) #(img_arr)

# Collect all images into a single numpy array
imgs = np.stack(listimgs)
# Extract each channel
R, G, B = imgs[:,:,:,0], imgs[:,:,:,1], imgs[:,:,:,2]
# Channel-wise mean value of whole dataset
means = np.mean(R), np.mean(G), np.mean(B)
# Channel-wise standard deviation value of whole dataset
stds = np.std(R), np.std(G), np.std(B)
print(means, stds)