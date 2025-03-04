from torch.utils.data import Dataset
import os
from netCDF4 import Dataset as nc_Dataset
import pandas as pd
import numpy as np

class MNISTNextDigitDataset(Dataset):
    """
    Assumes you've generated nc files as described in create_nc_MNIST.py and have mappings from MNIST_num2num_mapping.py.
    """
    def __init__(self, is_train = False):
        """
        :param is_train: bool - describes whether to load training or testing dataset
        """

        #get paths
        if is_train:
            data_save_path = os.path.join(os.getcwd(), "data", "MNIST", "nc_versions", "MNIST_training.nc")
            indices_save_path = os.path.join(os.getcwd(), "data", "MNIST", "num2num_indices", "train_indices.csv")
        else:
            data_save_path = os.path.join(os.getcwd(), "data", "MNIST", "nc_versions", "MNIST_testing.nc")
            indices_save_path = os.path.join(os.getcwd(), "data", "MNIST", "num2num_indices", "test_indices.csv")

        #open netCDF4 Dataset of MNIST
        self.nc_dataset = nc_Dataset(data_save_path)

        #read mapping indices from CSV
        self.predictor_indices, self.target_indices = self._read_mapping_indices_from_csv(indices_save_path)
        assert len(self.predictor_indices) == len(self.target_indices), "Predictor indices array should be of the same length as the target indices array"
        
    def __len__(self):
        return len(self.predictor_indices)
        
    def __getitem__(self, idx):

        #get sample index for predictor and target
        p_idx = self.predictor_indices[idx]
        t_idx = self.target_indices[idx]

        #extract images from nc_dataset
        predictor = self.nc_dataset["image"][p_idx,:,:].data[np.newaxis,:,:] #get image and add channel dimension
        target = self.nc_dataset["image"][t_idx,:,:].data[np.newaxis,:,:]
        
        return (predictor), (target) #important to return tuple of tuples! Allows nice X, y unpacking and allows PyTorch to do the tensor casting itself
    
    @staticmethod
    def _read_mapping_indices_from_csv(csv_path):
        df = pd.read_csv(csv_path)
        predictor_indices = df['predictor'].values 
        target_indices = df['target'].values
        return predictor_indices, target_indices