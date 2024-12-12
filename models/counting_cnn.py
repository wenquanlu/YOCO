import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from transformers import ViTModel


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(0.2)
        
        self.adjust_channels = None
        if in_channels != out_channels:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        if self.adjust_channels is not None:
            identity = self.adjust_channels(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity
        out = self.relu(out)

        return out


class CountingCNN(nn.Module):
    def __init__(self, num_deconv_layers=3):
        super(CountingCNN, self).__init__()
        self.num_deconv_layers = num_deconv_layers

        self.vit_extractor = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.resnet_encoder = nn.Sequential(
            ResidualBlock(768, 512),
            ResidualBlock(512, 256)
        )
        self.deconv_layrs = self.make_deconv_layer(num_deconv_layers, [256, 128, 64, 32])
        self.zero_conv = nn.Conv2d(32, 1, kernel_size=1)

    def make_deconv_layer(self, num_layers, num_filters):
        layers = []
        in_channel = num_filters[0]
        for i in range(num_layers):
            filters = num_filters[i]
            layers.append(nn.ConvTranspose2d(
                in_channels=in_channel,
                out_channels=filters,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))
            in_channel = filters
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.vit_extractor(x).last_hidden_state
        x = x[:, 1:, :]
        
        b, p, d = x.shape
        grid_size = int(np.sqrt(p))
        x = x.view(b, grid_size, grid_size, d)
        x = x.permute(0, 3, 1, 2)
        
        x = self.resnet_encoder(x)
        
        x = self.deconv_layrs(x)
        heatmap = self.zero_conv(x).squeeze(1)
        
        return heatmap

    def compute_loss(self, predicted_heatmaps, gt_heatmaps, sigma=2, lambda_keypoint=1.0, lambda_centered=1.0):
        """
        Compute the loss for heatmap prediction, with an additional loss term for keypoint regions 
        and a loss term that enforces higher values at keypoints and lower values around them.
        
        Parameters:
        - predicted_heatmaps: Tensor of shape (batch_size, H, W)
        - gt_heatmaps: Tensor of shape (batch_size, H, W)
        - sigma: Standard deviation for Gaussian kernel (controls the influence of the keypoint)
        - lambda_keypoint: Regularization factor for keypoint loss (emphasizing keypoints)
        - lambda_centered: Regularization factor for enforcing keypoint outputs higher than surrounding regions
        """
        basic_loss = F.mse_loss(predicted_heatmaps, gt_heatmaps, reduction='mean')
        keypoint_mask = (gt_heatmaps > 0).float()
        
        predicted_keypoint_values = predicted_heatmaps * keypoint_mask 
        mean_keypoint_value = predicted_keypoint_values.sum() / (keypoint_mask.sum() + 1e-6)
        
        keypoint_loss = torch.mean((predicted_heatmaps - mean_keypoint_value) * keypoint_mask)
        surrounding_max = F.max_pool2d(predicted_heatmaps.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        center_loss = torch.mean((surrounding_max - predicted_heatmaps) * keypoint_mask)
        
        total_loss = basic_loss + lambda_keypoint * keypoint_loss + lambda_centered * center_loss
        
        return total_loss