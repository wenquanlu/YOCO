import numpy as np
import torch
from data.dataset import WiderFaceDataset
from data.parse_dataset import parse_dataset
import argparse
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from counting_vit_cnn import CountingViTCNN
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam
import torch.nn.functional as F
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser("countingViT training")
    parser.add_argument("--train_set", default="wider_face_split/wider_face_train_bbx_gt.txt")
    parser.add_argument("--config_file", default="configs/config.yaml")
    parser.add_argument("--num_file", default=10)
    parser.add_argument("--state_dict", default="")
    return parser

def eval(args):
    data, labels = parse_dataset(args.train_set)
    train_dataset = WiderFaceDataset(data, labels, train=False)
    dataloader = DataLoader(train_dataset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=4, 
                            collate_fn=train_dataset.custom_collate_fn)
    model = CountingViTCNN(768)
    model.load_state_dict(torch.load(args.state_dict))
    model.cuda()
    model.eval()
    #i = 0
    counter = 0
    stop = False
    for imgs, coords, seq_lens, max_seq_len in dataloader:
        #if i == args.num_file:
        #    break
        if stop:
            break
        imgs = imgs.cuda()
        coords = coords.cuda()
        seq_lens = seq_lens.cuda()
        max_seq_len = max_seq_len.cuda()
        with torch.no_grad():
            predicted_heatmaps, predicted_coords = model(imgs, max_seq_len)
            predicted_heatmaps = predicted_heatmaps.cpu()

            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()  # Reshape for broadcasting
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()   # Reshape for broadcasting

            imgs = imgs * std + mean

            for k in range(predicted_heatmaps.shape[0]): # batch loop
                print(counter)
                if counter >= int(args.num_file):
                    print("stop!!!!!!!!!!!!")
                    stop = True
                    return
                image_tensor = imgs[k]
                
                # Clip values to [0, 1]
                clipped_tensor = torch.clamp(image_tensor, 0, 1)

                # Convert to NumPy array (H, W, C format)
                numpy_array = clipped_tensor.permute(1, 2, 0).cpu().numpy()

                # Convert to uint8 [0, 255]
                image_array = (numpy_array * 255).astype(np.uint8)

                # Save the image
                image = Image.fromarray(image_array)
                image.save(f"results/output_image_{counter}.jpg")  # Save each image with a unique name

                sub_counter = 0

                for j in range(predicted_heatmaps.shape[1]):
                    heatmap = predicted_heatmaps[k][j].numpy()
                    heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap)) * 255
                    heatmap = heatmap.astype(np.uint8)
                    grayscale_heatmap = Image.fromarray(heatmap, mode='L')
                    grayscale_heatmap.save("results/{}_{}_heatmap.jpg".format(counter, sub_counter))
                    sub_counter += 1

                counter += 1


if __name__== "__main__":
    args = get_args_parser().parse_args()
    eval(args)
