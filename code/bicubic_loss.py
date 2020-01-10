import numpy as np
import errors
import os

import spectral.io.envi as envi
from skimage.measure import compare_ssim  # to calculate ssim
from skimage.io import imsave, imread
from skimage.transform import resize



def read_fla_file(filename):
    fla_file = envi.open(filename + '.hdr', filename + '.fla')
    im = fla_file.load(scale=False)
    return np.float64(im)

def find_psnr(sr, hr):
    # 'Mean Squared Error'
    mse = np.sum((sr - hr) ** 2)
    mse /= float(sr.shape[0] * sr.shape[1] * sr.shape[2])
    PIXEL_MAX = 65535
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse + 1))
    return psnr

def normalise01(a):
    """
    normalise a 2D array to [0, 1]
    """    
    b = (a - a.min()) / (a.max() - a.min())    
    return b

psnr_metric = []

for i in np.arange(201, 221, 1):
    im_lr_valid = os.path.join(args.root_dir, "valid/image_{}_lr2".format(i))
    im_hr_valid = os.path.join(args.root_dir, "valid/image_{}_hr".format(i))
    im_lr = np.clip(read_fla_file(im_lr_valid), 0, 65535.0)
    im_hr = read_fla_file(im_hr_valid)
    im_lrx2 = resize(im_lr, (240,480), order=3)
    psnr_metric.append(find_psnr(im_lrx2, im_hr))

print("PSNR for validation data of track1: {}dB".format(np.mean(psnr_metric)))
