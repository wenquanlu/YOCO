import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import torchvision.transforms as T

class TrackableRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
        self.flipped = False  # Initialize a flag to track the flip
    
    def __call__(self, img):
        # Decide whether to flip
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
        ## read in coordinate here as a key value map between filename and coords

        ## read in filename as a list

    def __len__(self):
        # Return the number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get data and label for a given index
        img_file = self.data[idx]
        if self.train:
            img = Image.open("YOCO3k/train/images/" + img_file)
        else:
            img = Image.open("WIDER_val/images/" + img_file)
        label = self.labels[idx]

        # raw coords, label: [[xmin, ymin, width, height], ...] (seq_len, 4)
        # coords [[y, x], [y, x]...]
        coords = np.zeros((label.shape[0], 2))
        if len(coords) > 0:
            coords[:, 1] = label[:, 0] + (label[:, 2]-1)/2
            coords[:, 0] = label[:, 1] + (label[:, 3]-1)/2
            img, coords = self.resize_and_transform_coord(img, coords, self.train)
        else:
            img, coords = self.resize(img, self.train)
        # sample: (batch, 3, H, W)
        # coordinates: [batch, seq_len, 2]
        # heatmaps: ? [batch, seq_len, 1, H, W] (TYPICALLY GENERATED ON THE FLY)
        return img, coords

    # resize the img to 384x384
    # transform original coord (in list?)
    # coords will be used for finding adaptive arrangement 
    def resize_and_transform_coord(self, img, coords, train):
        width, height = img.size
        h_ratio = 384 / height
        w_ratio = 384 / width
        if train:
            transform = transforms.Compose([
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                    TrackableRandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),  # Convert to tensor
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            img = transform(img)
            flipped = transform.transforms[1].flipped
            if flipped:
                coords[:,1] = width - coords[:, 1] - 1
        else:
            transform = transforms.Compose([
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor(),  # Convert to tensor
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            img = transform(img)
        coords[:,0] = np.clip(coords[:,0] * h_ratio, 0, 383) # y coord
        coords[:,1] = np.clip(coords[:,1] * w_ratio, 0, 383) # x coord
        return img, coords

    def resize(self, img, train):
        if train:
            transform = transforms.Compose([
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                    TrackableRandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),  # Convert to tensor
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            img = transform(img)
        else:
            transform = transforms.Compose([
                    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor(),  # Convert to tensor
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
            img = transform(img)
        
        return img, np.array([])

    def custom_collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])  # Images
        seq_lens = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
        max_seq_len = torch.max(seq_lens) + 1 # the termination step
        coordinates = torch.ones((len(batch), max_seq_len, 2), dtype=torch.float) * 10000  # make other ground truth far away
        #coordinates = [item[1] for item in batch]  # List of variable-length tensors
        for i, item in enumerate(batch):
            seq_len = seq_lens[i]  # Current sequence length
            if len(item[1]) > 0:
                coordinates[i, :seq_len, :] = torch.tensor(item[1], dtype=torch.float)  # Copy label into coordinates
        return images, coordinates, seq_lens, max_seq_len



    



