"""
Training utilities with Classifier-Free Guidance (CFG) support.

This module extends the standard training utilities with CFG conditional dropout.
The key modification is in train_one_epoch_cfg, which randomly drops class conditioning
during training to enable the model to learn both conditional and unconditional generation.
"""

from typing import Optional, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import make_grid
from PIL import Image
from pathlib2 import Path
import yaml


def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg


def train_one_epoch_cfg(trainer, loader, optimizer, device, epoch, num_classes, cfg_dropout_prob=0.1):
    """
    Training function with Classifier-Free Guidance dropout.

    This function implements the core CFG training strategy:
    1. With probability p_cfg (cfg_dropout_prob), replace the true class embedding with
       the null embedding (all zeros)
    2. With probability 1 - p_cfg, use the true class embedding

    This enables the model to learn:
    - ε_θ(x_t, t, c): Conditional noise prediction
    - ε_θ(x_t, t, ∅): Unconditional noise prediction (∅ = all zeros)

    Parameters:
        trainer: GaussianDiffusionTrainer instance
        loader: DataLoader for training data
        optimizer: Optimizer instance
        device: Device to train on (cuda/cpu)
        epoch: Current epoch number
        num_classes: Number of classes for one-hot encoding
        cfg_dropout_prob: Probability of dropping class conditioning (p_cfg)
            Typical values: 0.1 (10%) to 0.2 (20%)

    Returns:
        Average training loss for the epoch
    """
    trainer.train()
    total_loss, total_num = 0., 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        for images, labels in data:
            optimizer.zero_grad()

            x_0 = images.to(device)

            # Create one-hot encoding for class labels
            y = F.one_hot(labels.to(device=device, dtype=torch.long),
                         num_classes=num_classes).float()

            # === CFG CONDITIONAL DROPOUT ===
            # Randomly select samples to drop conditioning
            batch_size = y.shape[0]
            dropout_mask = torch.rand(batch_size, device=device) < cfg_dropout_prob

            # Replace with null embedding (all zeros) for dropped samples
            y[dropout_mask] = 0.0  # Null embedding: [0, 0, 0, ..., 0]

            # Standard diffusion training loss
            loss = trainer(x_0, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_num += x_0.shape[0]

            # Track dropout statistics
            num_dropped = dropout_mask.sum().item()
            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": total_loss / total_num,
                "cfg_drop%": f"{num_dropped/batch_size*100:.1f}"
            })

    return total_loss / total_num


def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image into a picture.

    Parameters:
        images: a tensor with shape (batch_size, channels, height, width).
        nrow: decide how many images per row. Default `8`.
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def save_sample_image(images: torch.Tensor, show: bool = True, path: Optional[str] = None,
                      format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image including intermediate process into a picture.

    Parameters:
        images: images including intermediate process,
            a tensor with shape (batch_size, sample, channels, height, width).
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5

    grid = []
    for i in range(images.shape[0]):
        # for each sample in batch, concat all intermediate process images in a row
        t = make_grid(images[i], nrow=images.shape[1], **kwargs)  # (channels, height, width)
        grid.append(t)
    # stack all merged images to a tensor
    grid = torch.stack(grid, dim=0)  # (batch_size, channels, height, width)
    grid = make_grid(grid, nrow=1, **kwargs)  # concat all batch images in a different row, (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid
