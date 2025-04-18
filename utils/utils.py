import os
import torch
from matplotlib import pyplot as plt

def save_model_checkpoint(epoch, model, optimizer, loss, model_name = "dense_nn_2HL"):
    """
    Saves a model such that you can load it and resume training.
    :param epoch: int - the training epoch in which the model is being saved
    :param model: torch.nn.Module - your model you're training
    :param optimizer: torch.optim.Optimizer - the optimizer you're using to update nn params
    :param loss: object returned by your loss function (e.g. loss = loss_function(pred, y))
    :param model_name: str - name of the directory in which your model checkpoints will be saved
    """
    #save model for resuming training
    package_root = os.path.dirname(os.getcwd())
    model_save_dir = os.path.join(package_root, "models", "MNIST_num2num", "model_checkpoints", model_name)
    os.makedirs(model_save_dir, exist_ok = True)
    torch.save({
        "epoch" : epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
        }, os.path.join(model_save_dir, f"epoch{epoch}.pth"))

def evaluate_model(model, loss_function, test_dataloader):
    """
    Prints the average value of your model's loss on the test dataset
    :param model: torch.nn.Module - your model you're training
    :param loss_function: torch.nn.Module - your loss function
    :param test_dataloader: torch.utils.data.DataLoader - Your test dataset's data loader
    """
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for X, y in test_dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
    print(f"Avg test loss: {test_loss / len(test_dataloader)}")

def plot_num2num_prediction(X, y, pred):
    """
    X, y, and pred are all numpy arrays of dimension (28, 28)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(X.squeeze(), cmap="gray", vmin = 0., vmax = 1.)
    axes[0].set_title(f"Predictor Image")
    axes[0].axis("off")
    axes[1].imshow(pred.squeeze(), cmap="gray", vmin = 0., vmax = 1.)
    axes[1].set_title(f"Predicted Image")
    axes[1].axis("off")
    axes[2].imshow(y.squeeze(), cmap="gray", vmin = 0., vmax = 1.)
    axes[2].set_title(f"True Image")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()