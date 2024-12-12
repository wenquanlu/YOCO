import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T


class TrackableRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
        self.flipped = False
    
    def __call__(self, img):
        if random.random() < self.p:
            self.flipped = True
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            self.flipped = False
            return img


class WiderFaceDataset(Dataset):
    def __init__(self, data, labels, train=True):
        """
        Args:
            data (list/array): Your input data.
            labels (list/array): Corresponding labels.
            transform (callable, optional): Optional transform to apply to data.
        """
        self.data = data
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file = self.data[idx]
        
        if self.train:
            o_img = Image.open("datasets/YOCO3k/train/images/" + img_file)
        else:
            o_img = Image.open("datasets/YOCO3k/val/images/" + img_file)
        label = self.labels[idx]

        coords = np.zeros((label.shape[0], 2))

        coords[:, 1] = label[:, 0] + (label[:, 2]-1)/2
        coords[:, 0] = label[:, 1] + (label[:, 3]-1)/2

        label[:,2] += (label[:, 0]-1)
        label[:,3] += (label[:, 1]-1)
        label = label[:, [0, 2, 1, 3]] #(xmin, xmax, ymin, ymax)

        img, coords, bbox = self.resize_and_transform_coord(o_img, coords, label, self.train)

        return o_img, img, coords, bbox

    def resize_and_transform_coord(self, img, coords, label, train):
        width, height = img.size
        h_ratio = 384 / height
        w_ratio = 384 / width

        if train:
            transform = transforms.Compose([
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                    TrackableRandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            img = transform(img)
            flipped = transform.transforms[1].flipped
            if flipped:
                coords[:,1] = width - coords[:, 1] - 1
                label[:,0] = width - label[:,0] - 1
                label[:,2] = width - label[:,2] - 1

        else:
            transform = transforms.Compose([
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            img = transform(img)
        
        coords[:,0] = np.clip(coords[:,0] * h_ratio, 0, 383) # y coord
        coords[:,1] = np.clip(coords[:,1] * w_ratio, 0, 383) # x coord

        label[:,:2] = label[:, :2] * w_ratio
        label[:,2:4] = label[:,2:4] * h_ratio

        return img, coords, label

    def custom_collate_fn(self, batch):
        o_images = [item[0] for item in batch]
        images = torch.stack([item[1] for item in batch])

        seq_lens = torch.tensor([len(item[2]) for item in batch], dtype=torch.long)
        max_seq_len = torch.max(seq_lens) + 1

        coordinates = torch.ones((len(batch), max_seq_len, 2), dtype=torch.float) * 10000
        bboxes = torch.ones((len(batch), max_seq_len, 4), dtype=torch.float) * 10000

        for i, item in enumerate(batch):
            seq_len = seq_lens[i]
            if len(item[2]) > 0:
                coordinates[i, :seq_len, :] = torch.tensor(item[2], dtype=torch.float)
                bboxes[i, :seq_len, :] = torch.tensor(item[3], dtype=torch.float)

        return o_images, images, coordinates, seq_lens, max_seq_len, bboxes