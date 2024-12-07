import numpy as np
import torch
from data.dataset_accuray import WiderFaceDatasetEvalAccuracy
from data.parse_dataset_eval import parse_eval_dataset
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from counting_vit_cnn import CountingViTCNN
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from train import create_balanced_subset
from collections import defaultdict
from test_data import test_data, test_labels


def get_args_parser():
    parser = argparse.ArgumentParser("countingViT training")
    parser.add_argument("--val_set", default="wider_face_split/wider_face_val_bbx_gt.txt")
    parser.add_argument("--config_file", default="configs/config.yaml")
    parser.add_argument("--state_dict", default="")
    parser.add_argument("--max_count", default=10)
    return parser

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

    return gaussian


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

# def compute_loss(predicted_heatmaps, heatmaps, predicted_coords, seq_lens, max_seq_len, alpha=0.1):

#     # Create a range of sequence indices
#     seq_indices = torch.arange(max_seq_len, device=seq_lens.device)

#     # Compare seq_indices with seq_lens expanded to (batch_size, max_seq_len)
#     # <= because we also supervise for the last heatmap - the termination, empty heatmap
#     heatmap_mask = (seq_indices.unsqueeze(0) <= seq_lens.unsqueeze(1)).int()

#     # Expand to the desired shape (batch, max_seq_len, 1, 1)
#     heatmap_mask = heatmap_mask.unsqueeze(-1).unsqueeze(-1)

#     mse_loss = F.mse_loss(heatmaps, predicted_heatmaps, reduction='none')
#     masked_mse = mse_loss * heatmap_mask
#     l2_loss = masked_mse.sum() / (heatmap_mask.sum() * 384 * 384)

#     #path_len_loss = path_length_loss(predicted_coords, seq_lens)
#     #print("heatmap.su()", heatmap_mask.sum())
#     #print("path_len_loss", path_len_loss)
#     #print("predicted heatmap", predicted_heatmaps)
#     #print("heatmaps", heatmaps)

#     return l2_loss #+ path_len_loss /(384*100)


def eval(args):
    print(args.state_dict)
    # data, labels = parse_eval_dataset(args.val_set, int(args.max_count))
    # data, labels = create_balanced_subset(int(args.max_count), data, labels)
    # print([len(_) for _ in labels])
    # train_dataset = WiderFaceDatasetEvalAccuracy(data, labels, 15)
    test_label = [np.array(_) for _ in test_labels]
    eval_dataset = WiderFaceDatasetEvalAccuracy(test_data, test_label, 10)

    dataloader = DataLoader(eval_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4, 
                            collate_fn=eval_dataset.custom_collate_fn)
    model = CountingViTCNN(768)
    model.load_state_dict(torch.load(args.state_dict)["model_state_dict"])
    model.cuda()
    model.eval()
    model_count = 0
    gt_count = 0
    acc = 0

    model_count_by_class = defaultdict(int)
    gt_count_by_class = defaultdict(int)
    acc_by_class = defaultdict(int)

    for imgs, coords, seq_lens, max_seq_len, bboxes in tqdm(dataloader):
        imgs = imgs.cuda()
        seq_lens = seq_lens.cuda()
        gt_count += seq_lens[0]
        gt_count_by_class[int(seq_lens[0])] += seq_lens[0]
        with torch.no_grad():
            predicted_heatmaps, predicted_coords = model(imgs, max_seq_len)
            count = 0
            marker = np.zeros((len(bboxes[0])))
            for i in range(predicted_heatmaps.shape[1]):
                if torch.max(predicted_heatmaps[0][i]) > 4.0:
                    x = predicted_coords[0][i][1]
                    y = predicted_coords[0][i][0]
                    j = 0
                    for box in bboxes[0]:
                        xmin, xmax, ymin, ymax = box
                        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                            if marker[j] == 0:
                                acc += 1
                                acc_by_class[int(seq_lens[0])] += 1
                                #print("acc!!!!!!")
                                marker[j] = 1
                                break
                        j += 1
                    count += 1
                # else:
                #     break
        model_count_by_class[int(seq_lens[0])] += count
        model_count += count

        # print("################")
        # print(gt_count)
        # print(model_count)
        # print("################")

            #coords = rearrange_coords(predicted_coords, coords)
            # heatmaps is a tensor
            #heatmaps = add_gaussians_to_heatmaps_batch(predicted_heatmaps, coords)
            #print(torch.max(heatmaps), torch.min(heatmaps), "heatmaps")
            #with torch.autograd.detect_anomaly():
            #print("predicted_coords", predicted_coords)
            #print("rearranged coords", coords)
            #loss = compute_loss(predicted_heatmaps, heatmaps, predicted_coords, seq_lens, max_seq_len).cpu()
            #losses.append(loss)
    #print(gt_count)
    #print(model_count)
    overall_precision = acc/model_count
    overall_recall = acc/gt_count
    print("Precision:", overall_precision)
    print("Recall:", overall_recall)
    print("F-1 score:", 2*(overall_precision * overall_recall)/(overall_precision + overall_recall))
    #f = open("12_sorted_freezed_accuracy.txt", "a")
    #f.write(args.state_dict + ":" + str(overall_precision) + ", " + str(overall_recall) + ", "+ str(2*(overall_precision * overall_recall)/(overall_precision + overall_recall)))
    #f.write("\n")

    print(model_count_by_class)
    for key in gt_count_by_class:
        print("######################################")
        print("class: ", key)
        precision = acc_by_class[key]/model_count_by_class[key]
        recall = acc_by_class[key]/gt_count_by_class[key]
        f1 = 2*(precision * recall)/(precision + recall)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-1:", f1)








if __name__== "__main__":
    args = get_args_parser().parse_args()
    eval(args)
