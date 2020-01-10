# Evaluation script for the PRIM2018 Spectral super resolution Challenge
#
# * Provide input and output directories as arguments
# * Validation files should be found in the '/ref' subdirectory of the input dir
# * Input validation files are expected .fla format


import numpy as np
import sys
import os

import spectral.io.envi as envi
import spectral
spectral.settings.envi_support_nonlowercase_params = True
 # module to read ENVI images in python
from skimage.measure import compare_ssim  # to calculate ssim




MRAEs = {}
MSEs = {}
SSIMs = {}
SIDs = {}
APPSAs = {}
PSNRs = {}


#root_dir = "/mnt/md0/CSIRO/projects/2018_May_ECCV_challange/scoring_program/program_track1"
#os.chdir(root_dir)

def get_ref_from_file(filename):
    fla_file = envi.open(filename + '.hdr', filename + '.fla')
    im = fla_file.load(scale=False)
    return im

def mse(imageA, imageB):
	# 'Mean Squared Error'
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
	return err
 
def find_psnr(imageA, imageB):
	# 'Mean Squared Error'
    mse = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    PIXEL_MAX = 65536
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse + 1e-3))
    return psnr
 
def find_sid(gt, rc):
    N = gt.shape[2]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(rc[:,:,i] * np.log10((rc[:,:,i] + 1e-3)/(gt[:,:,i] + 1e-3))) +
                     np.sum(gt[:,:,i] * np.log10((gt[:,:,i] + 1e-3)/(rc[:,:,i] + 1e-3))))
    return err / (gt.shape[1] * gt.shape[0])
    
def find_appsa(gt,rc):
    
    nom = np.sum(gt * rc, axis=2)
    denom = np.linalg.norm(gt, axis=2) * np.linalg.norm(rc, axis=2)
    
    cos = np.where((nom/(denom + 1e-3)) > 1, 1, (nom/(denom + 1e-3)))
    appsa = np.arccos(cos)
        
    return np.sum(appsa)/(gt.shape[1] * gt.shape[0])
    
# input and output directories given as arguments
[_, input_dir, output_dir] = sys.argv
#####
#input_dir = "input"
#output_dir = "output"

validation_files = os.listdir(input_dir +'/ref')

for f in validation_files:
    # Read ground truth data
    if not(os.path.splitext(f)[1] in '.fla'):
        print('skipping '+f)
        continue
    gt = get_ref_from_file(input_dir + '/ref/' + os.path.splitext(f)[0])
    # Read user submission
    rc = get_ref_from_file(input_dir + '/res/' + os.path.splitext(f)[0])
    # compute MRAE
    diff = gt - rc
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff,gt + 1) # added epsilon to avoid division by zero.
    MRAEs[f] = np.mean(relative_abs_diff)
    print("f:")
    print(f)
    print("MRAEs[f]:")
    print(MRAEs[f])
    # compute SID
    SIDs[f] = find_sid(gt, rc)
    print("SIDs[f]:")
    print(SIDs[f])
    # compute Mean Squared Error'
    MSEs[f] = mse(gt, rc)
    print("MSEs[f]:")
    print(MSEs[f])
    # calculate ssim
    SSIMs[f] = compare_ssim(gt, rc)
    print("SSIMs[f]:")
    print(SSIMs[f])
    # calculate appsa
    APPSAs[f] = find_appsa(gt, rc)
    print("APPSAs[f]:")
    print(APPSAs[f])
    # calculate PSNR
    PSNRs[f] = find_psnr(gt, rc)
    print("PSNRs[f]:")
    print(PSNRs[f])
    

MRAE = np.mean(list(MRAEs.values()))
MSE = np.mean(list(MSEs.values()))
SSIM = np.mean(list(SSIMs.values()))
SID = np.mean(list(SIDs.values()))
APPSA = np.mean(list(APPSAs.values()))
PSNR = np.mean(list(PSNRs.values()))
print("MRAE:\n"+MRAE.astype(str))
print("MSE:\n"+MSE.astype(str))
print("SSIM:\n"+SSIM.astype(str))
print("SID:\n"+SID.astype(str))
print("APPSA:\n"+APPSA.astype(str))
print("PSNR:\n"+PSNR.astype(str))



with open(output_dir + '/scores.txt', 'w') as output_file:
    # write MRAE in score.txt
    output_file.write("MRAE:"+MRAE.astype(str))
    # write MSE in score.txt
    output_file.write("\nMSE:"+MSE.astype(str))
    # write SSIM in score.txt
    output_file.write("\nSSIM:"+SSIM.astype(str))
    # write SID in score.txt
    output_file.write("\nSID:"+SID.astype(str))
    # write SID in score.txt
    output_file.write("\nAPPSA:"+APPSA.astype(str))
    # write SID in score.txt
    output_file.write("\nPSNR:"+PSNR.astype(str))