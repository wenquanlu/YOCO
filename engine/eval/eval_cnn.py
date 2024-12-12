import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from data.dataset import WiderFaceDataset
from models.counting_cnn import CountingCNN
from utils.cnn_utils import compute_acc, find_objects

def eval_cnn(args):
    with open(args.val_set, 'r') as f:
        eval_set = json.load(f)
    eval_data, eval_labels = eval_set["data"], eval_set["labels"]

    eval_labels_np = [np.array(_) for _ in eval_labels]
    eval_dataset = WiderFaceDataset(eval_data, eval_labels_np, train=False)
    eval_dataloader = DataLoader(eval_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4, 
                            collate_fn=eval_dataset.custom_collate_fn)
    
    model = CountingCNN(num_deconv_layers=4)
    model.load_state_dict(torch.load(args.state_dict)["model_state_dict"])
    model.cuda()
    model.eval()
    
    index = 0
    acc_all = 0
    gt_count = 0
    model_count = 0
    acc_by_class = defaultdict(int)
    gt_count_by_class = defaultdict(int)
    model_count_by_class = defaultdict(int)
    
    for o_imgs, imgs, coords, seq_lens, max_seq_len, bbox in tqdm(eval_dataloader):
        imgs = imgs.cuda()
        coords = coords.cuda()
        seq_lens = seq_lens.cuda()
        max_seq_len = max_seq_len.cuda()
        with torch.no_grad():
            predicted_heatmaps = model(imgs)
            predicted_heatmaps, pred_coords, counts = find_objects(predicted_heatmaps)
        
        for i in range(len(imgs)):
            model_count += counts[i]
            gt_count += seq_lens[i]
            gt_count_by_class[int(seq_lens[i])] += seq_lens[i]
            model_count_by_class[int(seq_lens[i])] += counts[i]
            acc = compute_acc(bbox[i][:seq_lens[i]], pred_coords[i], )

            acc_all += acc
            acc_by_class[int(seq_lens[i])] += acc
        
        if args.visualize:
            o_imgs = [np.array(o_img.resize((384, 384))) for o_img in o_imgs]
            for i in range(len(o_imgs)):
                img = o_imgs[i]
                heatmap = predicted_heatmaps[i].cpu().numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                heatmap = plt.cm.jet(heatmap)[:, :, :3]
                img = img / 255.0
                overlay = (0.4 * img + 0.6 * heatmap).clip(0, 1)
                heatmap_path = os.path.join(args.vis_dir, f"heatmap_img_{index}.jpeg")

                plt.imshow(overlay)
                plt.title(f"GT: {seq_lens[i]}, PD: {counts[i]}")
                index += 1
                plt.savefig(heatmap_path)
                plt.close()

    for key in gt_count_by_class:
        p = acc_by_class[key]/model_count_by_class[key]
        r = acc_by_class[key]/gt_count_by_class[key]
        f1 = 2*(p * r)/(p + r)
        print("--------------------------------------")
        print("Class:", key)
        print("Precision:", p)
        print("Recall:", r)
        print("F1 score:", f1)

    overall_precision = acc_all/model_count
    overall_recall = acc_all/gt_count
    F1_score = 2*(overall_precision * overall_recall)/(overall_precision + overall_recall)
    print("--------------------------------------")
    print("Overall")
    print("Precision:", overall_precision)
    print("Recall:", overall_recall)
    print("F1 score:", F1_score)