import torch
import numpy as np
from scipy.ndimage import maximum_filter


def compute_acc(bbox, pred_coords):
    acc = 0
    matched_boxes = torch.zeros(bbox.shape[0], dtype=torch.bool, device=bbox.device)  # Track matched boxes
    
    # import pdb; pdb.set_trace()
    for coord in pred_coords:
        for i, box in enumerate(bbox):
            if matched_boxes[i]:  # Skip already matched boxes
                continue
            if box[0] <= coord[1] <= box[1] and box[2] <= coord[0] <= box[3]:
                acc += 1
                matched_boxes[i] = True  # Mark the box as matched
                break

    return acc

def nms(heatmap, threshold, window_size=3):
    """
    Apply Non-Maximum Suppression (NMS) to a heatmap.
    """
    max_filtered = maximum_filter(heatmap, size=window_size, mode='constant')
    peaks = (heatmap == max_filtered) & (heatmap > threshold)
    return np.argwhere(peaks)


def find_objects(heatmaps):
    B, H, W = heatmaps.shape
    coords_list = []
    count_list = []
    final_heatmaps = torch.zeros_like(heatmaps)
    
    for b in range(B):
        heatmap = heatmaps[b]
        mean_val = torch.mean(heatmap)
        std_val = torch.std(heatmap)
        lower_threshold = mean_val + 1 * std_val
        
        binary_map = (heatmap > lower_threshold).float()
        
        final_heatmap = heatmap * binary_map 
        final_heatmaps[b] = final_heatmap 
        
        norm_heatmap = (final_heatmap - final_heatmap.min()) / (final_heatmap.max() - final_heatmap.min())
        norm_heatmap_np = norm_heatmap.detach().cpu().numpy()
        keypoints = nms(norm_heatmap_np, threshold=0.3, window_size=30)

        coords = keypoints if len(keypoints) > 0 else np.empty((0, 2), dtype=np.float32)
        coords_list.append(coords)
        count_list.append(len(coords))
        
    
    return final_heatmaps, coords_list, count_list