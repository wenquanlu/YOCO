import numpy as np
import torch
from data.dataset_accuray import WiderFaceDatasetEvalAccuracy
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from counting_vit_cnn import CountingViTCNN
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam
import torch.nn.functional as F
from PIL import Image
from test_data import test_data, test_labels
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser("countingViT training")
    parser.add_argument("--config_file", default="configs/config.yaml")
    parser.add_argument("--state_dict", default="")
    return parser

def eval(args):
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
    counter = 0

    for imgs, coords, seq_lens, max_seq_len, bboxes in dataloader:
        imgs = imgs.cuda()
        coords = coords.cuda()
        seq_lens = seq_lens.cuda()
        with torch.no_grad():
            predicted_heatmaps, predicted_coords = model(imgs, max_seq_len)
            predicted_heatmaps = predicted_heatmaps.cpu()

            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()  # Reshape for broadcasting
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()   # Reshape for broadcasting

            imgs = imgs * std + mean

            for k in range(predicted_heatmaps.shape[0]): # batch loop
                print(counter)
                image_tensor = imgs[k]
                
                # Clip values to [0, 1]
                clipped_tensor = torch.clamp(image_tensor, 0, 1)

                # Convert to NumPy array (H, W, C format)
                img = clipped_tensor.permute(1, 2, 0).cpu().numpy()

                sub_counter = 0

                for j in range(predicted_heatmaps.shape[1]):
                    if torch.max(predicted_heatmaps[k][j]) > 4:
                        heatmap = predicted_heatmaps[k][j].numpy()
                        heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                        
                        heatmap = plt.cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
                        overlay = (0.4 * img + 0.6 * heatmap).clip(0, 1)
                        heatmap_path = "results/{}_{}_heatmap.jpg".format(counter, sub_counter)
                        plt.axis('off')

                        plt.imshow(overlay)
                        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        sub_counter += 1
                    else:
                        break
                counter += 1


if __name__== "__main__":
    args = get_args_parser().parse_args()
    eval(args)
