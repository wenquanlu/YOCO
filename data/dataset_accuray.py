import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import Image


class WiderFaceDatasetEvalAccuracy(Dataset):
    def __init__(self, data, labels, max_seq_len):
        """
        Args:
            data (list/array): Your input data.
            labels (list/array): Corresponding labels.
            transform (callable, optional): Optional transform to apply to data.
        """
        self.data = data
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        sample: (batch, 3, H, W)
        coordinates: [batch, seq_len, 2]
        heatmaps: ? [batch, seq_len, 1, H, W] (TYPICALLY GENERATED ON THE FLY)
        """
        img_file = self.data[idx]

        img = Image.open("datasets/YOCO3k/val/images/" + img_file)
        label = self.labels[idx]

        # raw coords, label: [[xmin, ymin, width, height], ...] (seq_len, 4)
        # coords [[y, x], [y, x]...]
        coords = np.zeros((label.shape[0], 2))

        coords[:, 1] = label[:, 0] + (label[:, 2]-1)/2
        coords[:, 0] = label[:, 1] + (label[:, 3]-1)/2

        label[:,2] += (label[:, 0]-1)
        label[:,3] += (label[:, 1]-1)
        label = label[:, [0, 2, 1, 3]] #(xmin, xmax, ymin, ymax)
        img, coords, bbox = self.resize_and_transform_coord(img, coords, label)

        return img, coords, bbox

    def resize_and_transform_coord(self, img, coords, label):
        width, height = img.size
        h_ratio = 384 / height
        w_ratio = 384 / width

        transform = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        img = transform(img)
        coords[:,0] = np.clip(coords[:,0] * h_ratio, 0, 383)
        coords[:,1] = np.clip(coords[:,1] * w_ratio, 0, 383)

        label[:,:2] = label[:, :2] * w_ratio
        label[:,2:4] = label[:,2:4] * h_ratio

        return img, coords, label

    def custom_collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        seq_lens = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
        max_seq_len = self.max_seq_len
        coordinates = torch.ones((len(batch), max_seq_len, 2), dtype=torch.float) * 10000
        bboxes = torch.ones((len(batch), max_seq_len, 4), dtype=torch.float) * 10000

        for i, item in enumerate(batch):
            seq_len = seq_lens[i]
            if len(item[1]) > 0:
                coordinates[i, :seq_len, :] = torch.tensor(item[1], dtype=torch.float)
                bboxes[i, :seq_len, :] = torch.tensor(item[2], dtype=torch.float)
                
        return images, coordinates, seq_lens, max_seq_len, bboxes



    



