import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from models.counting_rnn import CountingRNN
from data.dataset_accuray import WiderFaceDatasetEvalAccuracy


def eval_rnn(args):
    with open(args.val_set, 'r') as f:
        eval_set = json.load(f)
    eval_data, eval_labels = eval_set["data"], eval_set["labels"]

    eval_label_np = [np.array(_) for _ in eval_labels]
    eval_dataset = WiderFaceDatasetEvalAccuracy(eval_data, eval_label_np, 10)
    dataloader = DataLoader(eval_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4, 
                            collate_fn=eval_dataset.custom_collate_fn)
    
    model = CountingRNN(768)
    model.load_state_dict(torch.load(args.state_dict)["model_state_dict"])
    model.cuda()
    model.eval()

    index = 0
    acc = 0
    gt_count = 0
    model_count = 0
    acc_by_class = defaultdict(int)
    gt_count_by_class = defaultdict(int)
    model_count_by_class = defaultdict(int)

    for imgs, coords, seq_lens, max_seq_len, bboxes in tqdm(dataloader):
        index += 1
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
                                marker[j] = 1
                                break
                        j += 1
                    count += 1
        model_count_by_class[int(seq_lens[0])] += count
        model_count += count

        if args.visualize:
            predicted_heatmaps = predicted_heatmaps.cpu()

            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
            imgs = imgs * std + mean

            for k in range(predicted_heatmaps.shape[0]):
                image_tensor = imgs[k]
                clipped_tensor = torch.clamp(image_tensor, 0, 1)
                img = clipped_tensor.permute(1, 2, 0).cpu().numpy()

                sub_counter = 0

                for j in range(predicted_heatmaps.shape[1]):
                    if torch.max(predicted_heatmaps[k][j]) > 4:
                        heatmap = predicted_heatmaps[k][j].numpy()
                        heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))
                        
                        heatmap = plt.cm.jet(heatmap)[:, :, :3]
                        overlay = (0.4 * img + 0.6 * heatmap).clip(0, 1)
                        heatmap_path = os.path.join(args.vis_dir, "{}_{}_heatmap.jpg".format(index, sub_counter))
                        plt.axis('off')

                        plt.imshow(overlay)
                        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        sub_counter += 1
                    else:
                        break

    overall_precision = acc/model_count
    overall_recall = acc/gt_count
    print("Precision:", overall_precision)
    print("Recall:", overall_recall)
    print("F-1 score:", 2*(overall_precision * overall_recall)/(overall_precision + overall_recall))

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