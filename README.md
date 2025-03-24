# PyTorch Image-to-Image Tutorial with MNIST

## Configuring Your Environment
Either use one of the two conda environment yamls or run the following pip command in your own environment:

```bash
pip install torch torchvision torchaudio accelerate lightning netCDF4 xarray matplotlib jupyterlab tqdm
```

## Getting Data

First, create a netCDF version of MNIST with **create_nc_MNIST.py**. Then create mappings from digits to the next one with **MNIST_num2num_mapping.py**.

## Tutorials

- **train_dense_nn_MNIST_num2num.ipynb** : covers the basic PyTorch workflow
- **train_dense_nn_Accelerate.ipynb**: covers HuggingFace Accelerate
- **train_dense_nn_Lightning.ipynb**: covers PyTorch Lightning
- **train_cnn_encoder.ipynb**: covers CNN encoder-decoder implementation
- **train_EDM_CorrDiff.ipynb**: covers generative prediction of residuals via EDM. Note that the non-commercial license applies to the functions in diffusion_utils.py; those functions are adapted from the official [EDM GitHub repository](https://github.com/NVlabs/edm/tree/main) and make implementation much easier.