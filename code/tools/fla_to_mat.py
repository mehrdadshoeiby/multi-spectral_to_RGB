# -*- coding: utf-8 -*-
"""
convert fla files to mat files
"""


    
hdf5storage.savemat(input_dir + "im_l_14d_3.mat", {"data": im_l_14d},
                    format='7.3', store_python_metadata=True)