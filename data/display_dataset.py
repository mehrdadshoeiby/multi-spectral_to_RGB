from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import h5py

def read_mat(im, varname='data'):
    f = h5py.File(im,'r') 
    im = f.get(varname) 
    im = np.array(im, float).T
    
    return im   

#root = args.root_dir +"data/train/3.tif"
#root_ms = args.root_dir +"data/train/3.mat"
#root_ms = '/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/main/data_raw/multispectral/0200.tif'
#root_ms = '/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/main/RCAN_v0/data/multispectral/0074.tif'
#im = imread(root_ms)
#im_ms = read_mat(root_ms)

root_dir = (r"/mnt/md0/CSIRO/projects/2019_01_colormatch_sr/"
                 r"main/RCAN_v6/")
                 


for i in range(2,3,1):
    root_ms = root_dir +"data/train/{}.mat".format(i)
    im_ms = read_mat(root_ms)
    print(im_ms)

#plt.figure(), plt.imshow(im)
plt.figure(), plt.imshow(np.sum(im_ms,axis=2))
for i in range(16):
    plt.figure(), plt.imshow(im_ms[0:4,0:4,i])

# diplay training dataset see if I can see all??
