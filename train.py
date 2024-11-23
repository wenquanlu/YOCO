import numpy as np
import torch
from data.dataset import WiderFaceDataset
from data.parse_dataset import parse_dataset
import argparse
from torch.utils.data import DataLoader

def get_args_parser():
    parser = argparse.ArgumentParser("countingViT training")
    parser.add_argument("--train_set", default="")

    return parser


def add_gaussians_to_images(image_size, coordinates, sigma=2):
    """
    Add Gaussian blobs centered at given coordinates to a sequence of black images.

    Args:
    - image_size (tuple): Size of each image (height, width).
    - coordinates (np.ndarray): Array of shape (seq_len, 2), each row is (x, y) coordinates for Gaussian centers.
    - sigma (float): Standard deviation of the Gaussian.

    Returns:
    - np.ndarray: A stack of images with Gaussian blobs of shape (seq_len, height, width).
    """
    seq_len = coordinates.shape[0]  # Number of images matches number of coordinate pairs
    height, width = image_size
    images = np.zeros((seq_len, height, width), dtype=np.float32)

    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)  # Create a grid of x and y values

    for i in range(seq_len):
        cy, cx = coordinates[i]  # Get the center for the current Gaussian
        gaussian = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2 * sigma**2))  # Gaussian formula
        images[i] = gaussian  # Assign the Gaussian to the corresponding image

    return images

def loss():
    # for each item in batch
        # adaptive loss
        # get model's output
        # sort coordinates using the output
        # construct heatmap
        # do the loss and apply mask

        # calculate sum of L2 between consecutive coords -- mask the L2
    pass
    # add up the loss

    # return loss

    

def train(args):
    data, labels = parse_dataset(args.train_set)
    train_dataset = WiderFaceDataset(data, labels)

    pass

if __name__== "__main__":
    args = get_args_parser().parse_args()
    train(args)