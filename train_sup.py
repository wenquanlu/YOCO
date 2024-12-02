import numpy as np
import torch
from data.dataset import WiderFaceDataset
#from data.parse_dataset import parse_dataset
from data.parse_dataset_yoco3k import parse_dataset_yoco3k
from data.parse_dataset_eval import parse_eval_dataset
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from counting_vit_cnn_sup import CountingViTCNNSup
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
import torch.nn.functional as F
import wandb
import random
from val_data import val_data
from val_data import val_labels

def get_args_parser():
    parser = argparse.ArgumentParser("countingViT training")
    parser.add_argument("--train_set", default="wider_face_split/wider_face_train_bbx_gt.txt")
    parser.add_argument("--val_set", default="wider_face_split/wider_face_val_bbx_gt.txt")
    parser.add_argument("--config_file", default="configs/config.yaml")
    parser.add_argument("--state_dict", default="")
    parser.add_argument("--max_count", default=None)

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


#def add_gaussians_to_images(image_size, coordinates, sigma=2):
#    """
#    Add Gaussian blobs centered at given coordinates to a sequence of black images.
#
#    Args:
#    - image_size (tuple): Size of each image (height, width).
#    - coordinates (np.ndarray): Array of shape (seq_len, 2), each row is (x, y) coordinates for Gaussian centers.#
#    - sigma (float): Standard deviation of the Gaussian.
#
#    Returns:
#    - np.ndarray: A stack of images with Gaussian blobs of shape (seq_len, height, width).
#    """
#    seq_len = coordinates.shape[0]  # Number of images matches number of coordinate pairs
#    height, width = image_size
#    images = np.zeros((seq_len, height, width), dtype=np.float32)
#
#    x = np.arange(width)
#    y = np.arange(height)
#    xv, yv = np.meshgrid(x, y)  # Create a grid of x and y values
#
#    for i in range(seq_len):
#        cy, cx = coordinates[i]  # Get the center for the current Gaussian
#        gaussian = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2 * sigma**2))  # Gaussian formula
#        images[i] = gaussian  # Assign the Gaussian to the corresponding image
#
#    return images

# coordinates: (batch, max_seq_len, H, W)
# heatmaps: (batch, max_seq_len, H, W)
"""def add_gaussians_to_heatmaps(heatmaps, coordinates, sigma=2):
    
    # Add Gaussian blobs centered at given coordinates to a sequence of black images.
    
    batch_size, max_seq_len, height, width = heatmaps.shape

    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)  # Create a grid of x and y values
    for i in range(batch_size):
        seq_len = coordinates[i].shape[0]  # Number of images matches number of coordinate pairs
        for j in range(seq_len):
            cy, cx = coordinates[i][j]  # Get the center for the current Gaussian
            gaussian = np.exp(-((xv - cx)**2 + (yv - cy)**2) / (2 * sigma**2))  # Gaussian formula
            heatmaps[i][j] = gaussian  # Assign the Gaussian to the corresponding image
    return heatmaps
"""
# heatmaps: (batch, )
# predicted_coords: ()

def add_gaussians_to_heatmaps_batch(predicted_heatmaps, coordinates, sigma=2):
    """
    Add Gaussian blobs centered at given coordinates to heatmaps, parallelizing the batch.
    
    Parameters:
    - predicted_heatmaps: Tensor of shape (batch, max_seq_len, H, W)
    - coordinates: Tensor of shape (batch, max_seq_len, 2) with (y, x) coordinates
    - sigma: Standard deviation of the Gaussian blob
    """
    with torch.no_grad():
        batch_size, max_seq_len, height, width = predicted_heatmaps.shape
        
        # Create a grid for the image dimensions
        y = torch.arange(height, device=predicted_heatmaps.device).view(1, 1, -1, 1)
        x = torch.arange(width, device=predicted_heatmaps.device).view(1, 1, 1, -1)
        
        # Extract coordinates and reshape for broadcasting
        cy = coordinates[..., 0].view(batch_size, max_seq_len, 1, 1)  # (batch, seq_len, 1, 1)
        cx = coordinates[..., 1].view(batch_size, max_seq_len, 1, 1)  # (batch, seq_len, 1, 1)
        
        # Compute Gaussian blobs for all batches and sequences
        gaussian = 100 * torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))  # (batch, seq_len, H, W)
        
        # ONLY IF NECESSARY !!!!!!!!!!!!!!!!! ############### @@@@@@@@@@@@@@@@@@
        # Mask areas outside of valid sequence lengths (if needed)
        # mask = (coordinates[..., 0] == 10000) & (coordinates[..., 1] == 10000)  # Identify these timesteps
        # heatmaps[mask] = 0  # Explicitly set them to black
        cumulative_heatmaps = torch.zeros(batch_size, max_seq_len + 1, height, width, device=predicted_heatmaps.device)
        for t in range(max_seq_len):
            cumulative_heatmaps[:, t + 1] = cumulative_heatmaps[:, t] + gaussian[:, t]

    return gaussian, cumulative_heatmaps


def rearrange_coords(predicted_coords, coords):
    with torch.no_grad():
        batch_size, seq_len, _ = predicted_coords.shape
        rearranged_coords = torch.zeros_like(coords)
        
        used_indices = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=coords.device)

        for i in range(seq_len):
            # Compute pairwise distances between predicted_coords[:, i, :] and all coords
            distances = torch.norm(
                coords - predicted_coords[:, i].unsqueeze(1), dim=-1
            )  # Shape: (batch_size, seq_len, seq_len)

            # Mask already selected indices
            distances.masked_fill_(used_indices, float('inf'))

            # Find the minimum distances and corresponding indices
            min_indices = torch.argmin(distances, dim=1)  # Shape: (batch_size,)

            # Rearrange coordinates for this step
            rearranged_coords[:, i] = coords[torch.arange(batch_size, device=coords.device), min_indices]
            
            # Mark the selected indices as used
            used_indices[torch.arange(batch_size, device=coords.device), min_indices] = True

        return rearranged_coords


def match_coords(predicted_coords, coords, padding_value=10000.0):
    """
    Match ground truth coordinates to predicted coordinates based on pairwise L2 distances,
    while ensuring padding coordinates remain at the end.

    Args:
        predicted_coords (torch.Tensor): Tensor of shape (batch_size, seq_len, 2), predicted coordinates.
        coords (torch.Tensor): Tensor of shape (batch_size, seq_len, 2), ground truth coordinates.
        padding_value (float): Value used to indicate padding coordinates. Default is 10000.

    Returns:
        torch.Tensor: Rearranged ground truth coordinates, shape (batch_size, seq_len, 2).
    """
    with torch.no_grad():
        predicted_coords = predicted_coords.float()
        coords = coords.float()

        batch_size, seq_len, _ = predicted_coords.shape
        rearranged_coords = torch.full_like(coords, padding_value)

        # Mask for non-padding coordinates
        is_not_padding = (coords != padding_value).all(dim=-1)  # Shape: (batch_size, seq_len)

        for b in range(batch_size):
            # Get valid (non-padding) predicted and ground truth coordinates
            valid_predicted = predicted_coords[b][is_not_padding[b]]
            valid_coords = coords[b][is_not_padding[b]]

            # Compute pairwise distance for valid coordinates
            if valid_coords.shape[0] > 0:
                distances = torch.cdist(valid_predicted.unsqueeze(0), valid_coords.unsqueeze(0), p=2).squeeze(0)

                # Hungarian algorithm or Greedy match based on sorted distances
                matched_indices = distances.view(-1).argsort()
                assigned_pred = torch.zeros(valid_predicted.shape[0], dtype=torch.bool)
                assigned_gt = torch.zeros(valid_coords.shape[0], dtype=torch.bool)

                for idx in matched_indices:
                    pred_idx = idx // valid_coords.shape[0]
                    gt_idx = idx % valid_coords.shape[0]

                    if not assigned_pred[pred_idx] and not assigned_gt[gt_idx]:
                        rearranged_coords[b, pred_idx] = valid_coords[gt_idx]
                        assigned_pred[pred_idx] = True
                        assigned_gt[gt_idx] = True

        return rearranged_coords

def sort_naive(coords):
    with torch.no_grad():
        coords_np = coords.cpu().numpy()

        sorted_coords = []
        for batch in coords_np:
            # Extract x and y separately
            x = batch[:, 1]
            y = batch[:, 0]
            # Use lexsort: Sort by y (primary key) and x (secondary key)
            indices = np.lexsort((y, x))
            sorted_coords.append(batch[indices])
        sorted_coords = np.array(sorted_coords)
        # Convert back to PyTorch tensor
        sorted_coords = torch.tensor(sorted_coords, dtype=coords.dtype, device=coords.device)
    return sorted_coords


def path_length_loss(predicted_coords, seq_lens):
    """
    Calculate the average path length per step for a batch of sequences.
    
    Args:
        predicted_coords: Tensor of shape (batch, max_seq_len, 2), predicted (x, y) coordinates.
        seq_lens: Tensor of shape (batch,), actual number of points in each sequence.

    Returns:
        Scalar loss representing the average path length per step across the batch.
    """
    # Compute the pairwise differences between consecutive coordinates
    deltas = predicted_coords[:, 1:] - predicted_coords[:, :-1]  # (batch, max_seq_len-1, 2)
    print(deltas)
    # Compute the Euclidean distance for each consecutive pair
    distances = torch.norm(deltas, dim=-1)  # (batch, max_seq_len-1)
    
    # Mask the distances based on seq_lens to ignore invalid points
    batch_size, max_seq_len = predicted_coords.size(0), predicted_coords.size(1)
    mask = torch.arange(max_seq_len - 1).expand(batch_size, -1).to(seq_lens.device) < (seq_lens - 1).unsqueeze(1)
    #print(mask)
    masked_distances = distances * mask  # (batch, max_seq_len-1)
    
    # Compute the sum of distances for each sequence and normalize by valid step count
    total_path_length = masked_distances.sum(dim=1)  # (batch,)
    valid_steps = (seq_lens - 1).clamp(min=1).float()  # Number of valid steps per sequence

    valid_sequences = seq_lens > 1
    avg_path_length_per_step = total_path_length / valid_steps  # (batch,)
    avg_path_length_per_step = avg_path_length_per_step[valid_sequences]

    #print("avg_path_length_per_step", avg_path_length_per_step)
    # Average over the batch
    loss = avg_path_length_per_step.mean()  # Scalar
    
    return loss

# heatmaps (batch, max_seq_len, H, W)
# predicted_heatmap (batch, max_seq_len, H, W)
# predicted_coords (batch, max_seq_len, 2)
# coords (batch, max_seq_len, 2)
def compute_loss(predicted_heatmaps, heatmaps, predicted_coords, predicted_cum_heatmaps, cum_heatmaps, seq_lens, max_seq_len, alpha=0.1):

    # Create a range of sequence indices
    seq_indices = torch.arange(max_seq_len, device=seq_lens.device)

    # Compare seq_indices with seq_lens expanded to (batch_size, max_seq_len)
    # <= because we also supervise for the last heatmap - the termination, empty heatmap
    heatmap_mask = (seq_indices.unsqueeze(0) <= seq_lens.unsqueeze(1)).int()

    # Expand to the desired shape (batch, max_seq_len, 1, 1)
    heatmap_mask = heatmap_mask.unsqueeze(-1).unsqueeze(-1)

    mse_loss = F.mse_loss(heatmaps, predicted_heatmaps, reduction='none')
    masked_mse = mse_loss * heatmap_mask
    cum_loss = F.mse_loss(cum_heatmaps, predicted_cum_heatmaps, reduction='none')
    masked_cum_mse = cum_loss * heatmap_mask
    l2_loss = (masked_mse.sum()+masked_cum_mse.sum()) / (2 * heatmap_mask.sum() * 384 * 384)

    #path_len_loss = path_length_loss(predicted_coords, seq_lens)
    print("l2_loss", l2_loss)
    #print("heatmap.su()", heatmap_mask.sum())
    #print("path_len_loss", path_len_loss)
    #print("predicted heatmap", predicted_heatmaps)
    #print("heatmaps", heatmaps)

    return l2_loss #+ path_len_loss /(384*100)

def create_balanced_subset(max_count, data, labels):
    baskets = [[] for _ in range(max_count)]

    for i in range(len(labels)):
        if len(labels[i]) > 0 and len(labels[i]) <= max_count:
            baskets[len(labels[i])-1].append(i)
    min_len = np.min([len(_) for _ in baskets])
    
    data_subset = []
    labels_subset = []

    for basket in baskets:
        sample_indices = random.sample(basket,min_len)
        for index in sample_indices:
            data_subset.append(data[index])
            labels_subset.append(labels[index])
    return data_subset, labels_subset

def train(args):
    config = OmegaConf.load(args.config_file)
    wandb.init(
        project="yoco3k",  # Replace with your project name
        config={}
    )
    data, labels = parse_dataset_yoco3k(args.train_set)
    #if args.max_count is not None:
    #    data, labels = create_balanced_subset(int(args.max_count), data, labels)
    print("data_len:", len(data), "labels len:", len(labels))
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
    model = CountingViTCNNSup(768)
    if args.state_dict != "":
        print("load", args.state_dict)
        model.load_state_dict(torch.load(args.state_dict)["model_state_dict"])
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-6)
    iteration_per_epoch = len(dataloader)
    warmup_scheduler = WarmUpLR(optimizer, warmup_epochs * iteration_per_epoch, warmup_lr, base_lr)
    #cosine_scheduler = CosineAnnealingWarmRestarts(
    #    optimizer,
    #    T_0 = (config.training.epochs - warmup_epochs) * iteration_per_epoch, 
    #    T_mult = 1,
    #    eta_min=scheduler_config["eta_min"],
    #)

    iteration = 0

    for epoch in range(config.training.epochs):
        model.train()
        if epoch == warmup_epochs:
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max = config.training.epochs * iteration_per_epoch, 
                eta_min=scheduler_config["eta_min"],
            )

        for imgs, coords, seq_lens, max_seq_len in dataloader:
            # coords: [batch, max_seq_len, 2]
            # max_seq_len = max(seq_lens) + 1
            print(seq_lens)
            #print(max_seq_len)
            #print(coords.shape[0])
            #if torch.isnan(imgs).any() or torch.isinf(imgs).any():
            #    print("Invalid values in input data!")
            imgs = imgs.cuda()
            coords = coords.cuda()
            seq_lens = seq_lens.cuda()
            max_seq_len = max_seq_len.cuda()

            ## TODO
            predicted_heatmaps, predicted_coords, predicted_cum_heatmaps = model(imgs, max_seq_len)
            # we need to sort coords here!!!!!!!!!!!!
            coords = match_coords(predicted_coords, coords)
            print("predicted_coords", predicted_coords)
            print("coords", coords)
            # heatmaps is a tensor
            heatmaps, cum_heatmaps = add_gaussians_to_heatmaps_batch(predicted_heatmaps, coords)
            #print(torch.max(heatmaps), torch.min(heatmaps), "heatmaps")
            #with torch.autograd.detect_anomaly():
            loss = compute_loss(predicted_heatmaps, heatmaps, predicted_coords, predicted_cum_heatmaps, cum_heatmaps, seq_lens, max_seq_len)
            optimizer.zero_grad()
            loss.backward()
            print("curr_lr", optimizer.param_groups[0]['lr'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
            # Step cosine annealing scheduler after warm-up
                cosine_scheduler.step(iteration)
            wandb.log({"epoch": epoch, "iteration": iteration, "train_loss": loss, "LR": optimizer.param_groups[0]['lr']})
            print(loss)
            iteration += 1
        if epoch >= warmup_epochs:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': cosine_scheduler.state_dict()
            }
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
        torch.save(checkpoint, 'run_2/model_state_{}.pth'.format(epoch))
        ### eval
        model.eval()
        val_label = [np.array(_) for _ in val_labels]
        eval_dataset = WiderFaceDataset(val_data, val_label, train=False)
        eval_dataloader = DataLoader(eval_dataset, 
                                batch_size=config.training.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                collate_fn=eval_dataset.custom_collate_fn)
        losses = []
        for imgs, coords, seq_lens, max_seq_len in eval_dataloader:
            #if i == args.num_file:
            #    break
            imgs = imgs.cuda()
            coords = coords.cuda()
            seq_lens = seq_lens.cuda()
            max_seq_len = max_seq_len.cuda()
            with torch.no_grad():
                predicted_heatmaps, predicted_coords, _ = model(imgs, max_seq_len)
                coords = match_coords(predicted_coords, coords)
                # heatmaps is a tensor
                heatmaps = add_gaussians_to_heatmaps_batch(predicted_heatmaps, coords)
                #print(torch.max(heatmaps), torch.min(heatmaps), "heatmaps")
                #with torch.autograd.detect_anomaly():
                loss = compute_loss(predicted_heatmaps, heatmaps, predicted_coords, seq_lens, max_seq_len).cpu()
                losses.append(loss)
        wandb.log({"eval_loss": np.mean(losses), "eval_data_len": len(val_data)})
        pass


if __name__== "__main__":
    args = get_args_parser().parse_args()
    train(args)
