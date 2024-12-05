import torch

# predicted_coords: (batch, max_seq_len, 2)
# coords: (batch, max_seq_len, 2)
def rearrange_coords(predicted_coords, coords):
    batch_size, seq_len, _ = predicted_coords.shape
    rearranged_coords = torch.zeros_like(coords)
    
    # used_indices in coords
    used_indices = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=coords.device)

    for i in range(seq_len):
        # Compute pairwise distances between predicted_coords[:, i, :] and all coords

        distances = torch.norm(
            coords.unsqueeze(2) - predicted_coords[:, i].unsqueeze(1), dim=-1
        )  # Shape: (batch_size, seq_len, seq_len)
        # coords.unsqueeze(2) (batch, max_seq_len, 1, 2)
        # predicted_coords[:, i].unsqueeze(1) (batch, 1, 1, 2)

        # Mask already selected indices
        distances.masked_fill_(used_indices.unsqueeze(1), float('inf'))

        # Find the minimum distances and corresponding indices
        min_indices = torch.argmin(distances, dim=1)  # Shape: (batch_size,)

        # Rearrange coordinates for this step
        rearranged_coords[:, i] = coords[torch.arange(batch_size, device=coords.device), min_indices]
        
        # Mark the selected indices as used
        used_indices[torch.arange(batch_size, device=coords.device), min_indices] = True

    return rearranged_coords

# Example usage:
predicted_coords = torch.tensor(
    [[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]], dtype=torch.float32
)
coords = torch.tensor(
    [[[3, 3], [1, 1], [2, 2]], [[6, 6], [4, 4], [5, 5]]], dtype=torch.float32
)

rearranged_coords = rearrange_coords(predicted_coords, coords)
print(rearranged_coords)



def rearrange_coords_with_indices(predicted_coords, coords):
    batch_size, seq_len, _ = predicted_coords.shape
    rearranged_coords = torch.zeros_like(coords)
    rearrangement_indices = torch.zeros((batch_size, seq_len), dtype=torch.long, device=coords.device)
    
    used_indices = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=coords.device)

    for i in range(seq_len):
        # Compute pairwise distances between predicted_coords[:, i, :] and all coords
        distances = torch.norm(
            coords.unsqueeze(2) - predicted_coords[:, i].unsqueeze(1), dim=-1
        )  # Shape: (batch_size, seq_len, seq_len)

        # Mask already selected indices
        distances.masked_fill_(used_indices.unsqueeze(1), float('inf'))

        # Find the minimum distances and corresponding indices
        min_indices = torch.argmin(distances, dim=1)  # Shape: (batch_size,)

        # Store the indices for this step
        rearrangement_indices[:, i] = min_indices

        # Rearrange coordinates for this step
        rearranged_coords[:, i] = coords[torch.arange(batch_size, device=coords.device), min_indices]
        
        # Mark the selected indices as used
        used_indices[torch.arange(batch_size, device=coords.device), min_indices] = True

    return rearranged_coords, rearrangement_indices