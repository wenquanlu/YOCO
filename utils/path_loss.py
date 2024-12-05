import torch
import torch.nn.functional as F

def path_length_loss(predicted_coords, seq_lens):
    """
    Calculate the path length loss for a batch of sequences.
    
    Args:
        predicted_coords: Tensor of shape (batch, max_seq_len, 2), predicted (x, y) coordinates.
        seq_lens: Tensor of shape (batch,), actual number of points in each sequence.

    Returns:
        Scalar loss representing the total path length across the batch.
    """
    # Compute the pairwise differences between consecutive coordinates
    deltas = predicted_coords[:, 1:] - predicted_coords[:, :-1]  # (batch, max_seq_len-1, 2)
    
    # Compute the Euclidean distance for each consecutive pair
    distances = torch.norm(deltas, dim=-1)  # (batch, max_seq_len-1)
    
    # Mask the distances based on seq_lens to ignore invalid points
    batch_size, max_seq_len = predicted_coords.size(0), predicted_coords.size(1)
    mask = torch.arange(max_seq_len - 1).expand(batch_size, -1).to(seq_lens.device) < (seq_lens - 1).unsqueeze(1)
    masked_distances = distances * mask  # (batch, max_seq_len-1)
    
    # Sum the distances for each sequence and average over the batch
    total_path_length = masked_distances.sum(dim=1)  # (batch,)
    loss = total_path_length.mean()  # Scalar
    
    return loss


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
    
    # Compute the Euclidean distance for each consecutive pair
    distances = torch.norm(deltas, dim=-1)  # (batch, max_seq_len-1)
    
    # Mask the distances based on seq_lens to ignore invalid points
    batch_size, max_seq_len = predicted_coords.size(0), predicted_coords.size(1)
    mask = torch.arange(max_seq_len - 1).expand(batch_size, -1).to(seq_lens.device) < (seq_lens - 1).unsqueeze(1)
    masked_distances = distances * mask  # (batch, max_seq_len-1)
    
    # Compute the sum of distances for each sequence and normalize by valid step count
    total_path_length = masked_distances.sum(dim=1)  # (batch,)
    valid_steps = (seq_lens - 1).float()  # Number of valid steps per sequence
    avg_path_length_per_step = total_path_length / valid_steps  # (batch,)
    
    # Average over the batch
    loss = avg_path_length_per_step.mean()  # Scalar
    
    return loss
