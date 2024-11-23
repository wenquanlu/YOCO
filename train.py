import numpy as np
import torch
from data.dataset import WiderFaceDataset
from data.parse_dataset import parse_dataset
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from counting_vit import CountingViT
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam

def get_args_parser():
    parser = argparse.ArgumentParser("countingViT training")
    parser.add_argument("--train_set", default="")
    parser.add_argument("--confg_file", default="configs/config.yaml")

    return parser

class WarmUpLR:
    def __init__(self, optimizer, warmup_steps, initial_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr + (self.base_lr - self.initial_lr) * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_step += 1


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

def compute_loss():
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
    config = OmegaConf.load(args.config_file)
    data, labels = parse_dataset(args.train_set)
    train_dataset = WiderFaceDataset(data, labels)

    dataloader = DataLoader(train_dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            collate_fn=train_dataset.custom_collate_fn)

    scheduler_config = config.training.scheduler
    warmup_epochs = scheduler_config["warmup_epochs"]
    warmup_lr = scheduler_config["warmup_lr"]
    base_lr = scheduler_config["base_lr"]
    model = CountingViT(768)
    optimizer = Adam(model.parameters(), lr=0.001)
    warmup_scheduler = WarmUpLR(optimizer, warmup_epochs, warmup_lr, base_lr)
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = (config.training.epochs - warmup_epochs) * iteration_per_epoch, 
        T_mult = 1,
        eta_min=scheduler_config["eta_min"],
    )

    iteration = 0
    iteration_per_epoch = len(dataloader)

    for epoch in range(config.training.epochs):
        for imgs, coords in dataloader:
            ## TODO     
            heatmaps, predicted_coords = model(imgs)
            loss = compute_loss(heatmaps, predicted_coords, coords)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                # Step cosine annealing scheduler after warm-up
                cosine_scheduler.step(iteration)
            iteration += 1
        pass


if __name__== "__main__":
    args = get_args_parser().parse_args()
    train(args)