#!/usr/local/bin/python
import torch.utils.data as data
import torch
import h5py
import tables
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')
        print (type(self.data))

    def __getitem__(self, index):            
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]

class DatasetFromHdf5_memory(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5_memory, self).__init__()
        hf = tables.open_file(file_path, driver="H5FD_CORE")
        self.data = hf.root.data
        self.target = hf.root.label
        print (type(self.data))

    def __getitem__(self, index):            
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]
