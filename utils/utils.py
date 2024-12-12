import torch
import numpy as np
import random

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
        
    return gaussian

def add_gaussians_to_heatmaps_batch_cum(predicted_heatmaps, coordinates, sigma=2):
    """
    Add Gaussian blobs centered at given coordinates to heatmaps, parallelizing the batch.
    
    Parameters:
    - predicted_heatmaps: Tensor of shape (batch, max_seq_len, H, W)
    - coordinates: Tensor of shape (batch, max_seq_len, 2) with (y, x) coordinates
    - sigma: Standard deviation of the Gaussian blob
    """
    with torch.no_grad():
        batch_size, max_seq_len, height, width = predicted_heatmaps.shape
        
        y = torch.arange(height, device=predicted_heatmaps.device).view(1, 1, -1, 1)
        x = torch.arange(width, device=predicted_heatmaps.device).view(1, 1, 1, -1)
        
        cy = coordinates[..., 0].view(batch_size, max_seq_len, 1, 1)  # (batch, seq_len, 1, 1)
        cx = coordinates[..., 1].view(batch_size, max_seq_len, 1, 1)  # (batch, seq_len, 1, 1)
        
        gaussian = 100 * torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))  # (batch, seq_len, H, W)
        
        cumulative_heatmaps = torch.zeros(batch_size, max_seq_len, height, width, device=predicted_heatmaps.device)
        for t in range(max_seq_len-1):
            cumulative_heatmaps[:, t + 1] = cumulative_heatmaps[:, t] + gaussian[:, t]

    return gaussian, cumulative_heatmaps


def add_gaussians_to_heatmaps_solo(predicted_heatmaps, coordinates, sigma=2):
    with torch.no_grad():
        gt_heatmap = []
        batch_size, height, width = predicted_heatmaps.shape
        max_num_obj = coordinates.shape[1]
        
        y = torch.arange(height, device=predicted_heatmaps.device).view(1, -1, 1)
        x = torch.arange(width, device=predicted_heatmaps.device).view(1, 1, -1)

        for b in range(batch_size):
            heatmap = torch.zeros(height, width, device=predicted_heatmaps.device)
            
            for obj in range(max_num_obj):
                cy = coordinates[b, obj, 0].view(1, 1)
                cx = coordinates[b, obj, 1].view(1, 1)
                if cy >= 1000 or cx >= 10000:
                    break
                gaussian = 100 * torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                heatmap += gaussian.squeeze()

            gt_heatmap.append(heatmap)

        gt_heatmap = torch.stack(gt_heatmap, dim=0)

    return gt_heatmap


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
    
def sort_naive(coords):
    with torch.no_grad():
        coords_np = coords.cpu().numpy()

        sorted_coords = []
        for batch in coords_np:
            x = batch[:, 1]
            y = batch[:, 0]
            # Use lexsort: Sort by y (primary key) and x (secondary key)
            indices = np.lexsort((y, x))
            sorted_coords.append(batch[indices])
        sorted_coords = np.array(sorted_coords)
        sorted_coords = torch.tensor(sorted_coords, dtype=coords.dtype, device=coords.device)
    return sorted_coords

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

def save_checkpoint(model, optimizer, scheduler, epoch, warmup_epochs, save_dir):
    if epoch >= warmup_epochs:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
    save_dir = os.path.join(save_dir, 'model_state_{}.pth'.format(epoch))
    torch.save(checkpoint, save_dir)

    return