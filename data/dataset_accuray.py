import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import torchvision.transforms as T

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
        ## read in coordinate here as a key value map between filename and coords

        ## read in filename as a list

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get data and label for a given index
        img_file = self.data[idx]

        img = Image.open("WIDER_val/images/" + img_file)
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

        # sample: (batch, 3, H, W)
        # coordinates: [batch, seq_len, 2]
        # heatmaps: ? [batch, seq_len, 1, H, W] (TYPICALLY GENERATED ON THE FLY)
        return img, coords, bbox

    # resize the img to 384x384
    # transform original coord (in list?)
    # coords will be used for finding adaptive arrangement 
    def resize_and_transform_coord(self, img, coords, label):
        width, height = img.size
        h_ratio = 384 / height
        w_ratio = 384 / width

        transform = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        img = transform(img)
        coords[:,0] = np.clip(coords[:,0] * h_ratio, 0, 383) # y coord
        coords[:,1] = np.clip(coords[:,1] * w_ratio, 0, 383) # x coord
        label[:,:2] = label[:, :2] * w_ratio
        label[:,2:4] = label[:,2:4] * h_ratio

        return img, coords, label

    # def resize(self, img, train):
    #     if train:
    #         transform = transforms.Compose([
    #                 transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
    #                 TrackableRandomHorizontalFlip(p=0.5),
    #                 transforms.ToTensor(),  # Convert to tensor
    #                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #             ])
    #         img = transform(img)
    #     else:
    #         transform = transforms.Compose([
    #                 transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
    #                 transforms.ToTensor(),  # Convert to tensor
    #                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #             ])
    #         img = transform(img)
        
    #     return img, np.array([])

    def custom_collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])  # Images
        seq_lens = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
        max_seq_len = self.max_seq_len
        coordinates = torch.ones((len(batch), max_seq_len, 2), dtype=torch.float) * 10000  # make other ground truth far away
        #coordinates = [item[1] for item in batch]  # List of variable-length tensors
        bboxes = torch.ones((len(batch), max_seq_len, 4), dtype=torch.float) * 10000
        for i, item in enumerate(batch):
            seq_len = seq_lens[i]  # Current sequence length
            if len(item[1]) > 0:
                coordinates[i, :seq_len, :] = torch.tensor(item[1], dtype=torch.float)  # Copy label into coordinates
                bboxes[i, :seq_len, :] = torch.tensor(item[2], dtype=torch.float)
        return images, coordinates, seq_lens, max_seq_len, bboxes



    



