import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import xarray as xr
import os
import numpy as np

def convert_to_xr(dataset):
    """
    Takes the PyTorch tensrs from datasets.FashionMNIST and creates an xarray Dataset
    which can then be saved as a netCDF file. This is done so that I can create my
    own custom data loader.
    """
    # Extract images and labels
    images = []
    labels = []
    
    for img, label in dataset:
        images.append(img.squeeze().numpy())  # Squeeze to remove batch dimension
        labels.append(label)
    
    # Convert to NumPy arrays
    images = np.array(images)  # Shape: (num_samples, height, width)
    labels = np.array(labels)  # Shape: (num_samples,)
    
    # Create an xarray.Dataset
    xr_dataset = xr.Dataset(
        {
            "image": (["sample_idx", "row", "column"], images),
            "label": (["sample_idx"], labels),
        },
        coords={
            "sample_idx": np.arange(len(labels)),
            "row": np.arange(images.shape[1]),
            "column": np.arange(images.shape[2]),
        }
    )
    
    return xr_dataset

def main():
    
    #these have implemented torch Dataset class and can be looped over or indexed to get items
    #each element returned is a tuple of length 2
    #The first element is a torch tensor of dimension (channel, height, width)
    #The second element is an integer corresponding to the class label
    
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    #paths will be your working directory + data/FashionMNIST (created when downloading data) + ncversions/fname.nc
    train_save_path = os.path.join(os.getcwd(), "data", "MNIST", "nc_versions", "MNIST_training.nc")
    test_save_path = os.path.join(os.getcwd(), "data", "MNIST", "nc_versions", "MNIST_testing.nc")

    #create nc folder
    os.makedirs(os.path.join(os.getcwd(), "data", "MNIST", "nc_versions"), exist_ok = True)
    
    #convert to xr and save to netcdf files
    convert_to_xr(training_data).to_netcdf(train_save_path)
    convert_to_xr(test_data).to_netcdf(test_save_path)

if __name__ == "__main__":
    main()