import os
import numpy as np
np.random.seed(42)
import xarray as xr
import pandas as pd

def generate_indices(dataset, samples_per_digit=10_000, seed=42):
    """
    Given an MNIST netcdf dataset, find indices mapping from digit n to
    digit (n+1)%10. E.g. 4 -> 5

    Returns two np arrays of equal lengths containing indices along the
    sample dimension of the netCDF file.
    
    :param dataset: xarray dataset of an MNIST netcdf
    :kwarg samples_per_digit: Integer number of mappings to make for each digit
    :kwarg seed: int specifying the random seed for deterministic behavior
    """
    np.random.seed(seed)
    
    #dict mapping each digit to its indices
    idx_dict = {num: np.where(dataset["label"].values == num)[0] for num in range(10)}
    
    predictor_indices = []
    target_indices = []
    
    #create arrays of digit indices
    for num in range(10):
        predictor_indices.append(np.random.choice(idx_dict[num], size=samples_per_digit, replace=True))
        target_indices.append(np.random.choice(idx_dict[(num + 1) % 10], size=samples_per_digit, replace=True))
    predictor_indices = np.concatenate(predictor_indices)
    target_indices = np.concatenate(target_indices)
    
    return predictor_indices, target_indices

def main():

    #open MNIST netcdfs
    in_dir = os.path.join(os.getcwd(), "data", "MNIST", "nc_versions")
    train_ds = xr.open_dataset(os.path.join(in_dir, "MNIST_training.nc"))
    test_ds = xr.open_dataset(os.path.join(in_dir, "MNIST_testing.nc"))

    #generate mappings
    train_predictor_indices, train_target_indices = generate_indices(train_ds, samples_per_digit=10_000)
    test_predictor_indices, test_target_indices = generate_indices(test_ds, samples_per_digit=2000)

    #convert mapping arrays to pandas dataframes (for ez csv saving)
    train_df = pd.DataFrame({'predictor': train_predictor_indices, 'target': train_target_indices})
    test_df = pd.DataFrame({'predictor': test_predictor_indices, 'target': test_target_indices})

    #save to csv
    save_dir = os.path.join(os.getcwd(), "data", "MNIST", "num2num_indices")
    os.makedirs(save_dir, exist_ok=True)  
    train_df.to_csv(os.path.join(save_dir, "train_indices.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test_indices.csv"), index=False)

if __name__ == "__main__":
    main()