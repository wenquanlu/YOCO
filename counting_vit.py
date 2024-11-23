import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
from models.vit import Transformer
import torch.nn.functional as F


vit2_configs = {
    "dim": 768,
    "depth": 4,
    "heads": 12,
    "dim_head": 64,
    "mlp_dim": 3072
}

class CountingViT(nn.Module):

    def __init__(self, hidden_dim, lstm_hidden_dim, vit2_configs, num_deconv_layers=4):
        
        self.vit_extractor = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.vit2 = Transformer(vit2_configs["dim"], 
                                vit2_configs["depth"], 
                                vit2_configs["heads"], 
                                vit2_configs["dim_head"],
                                vit2_configs["mlp_dim"])
        self.num_deconv_layers = num_deconv_layers
        self.deconv_layrs = self.make_deconv_layer(num_deconv_layers, [256, 256, 128, 64])
        self.zero_conv = nn.Conv2d(64, 1, 1)

    # reference to mmpose deconv_head
    def make_deconv_layer(self, num_layers, num_filters):
        layers = []
        in_channel = 768
        for i in range(num_layers):
            filters = num_filters[i]
            deconv_layer = nn.ConvTranspose2d(
                in_channels=in_channel,   # Input channels from ViT
                out_channels=num_filters[i],    # Single-channel heatmap output
                kernel_size=4,     # Kernel size
                stride=2,          # Upsampling factor (doubles spatial dimensions)
                padding=1          # Padding to match desired output size
            )     
            layers.append(deconv_layer)
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))   
            in_channel = num_filters[i]
        return nn.Sequential(*layers)

    def soft_argmax_2d(self, heatmap):
        # Get the shape
        B, H, W = heatmap.shape
        
        # Apply softmax to normalize the heatmap
        heatmap_flat = heatmap.view(B, -1)  # Flatten to (B, H*W)
        heatmap_norm = F.softmax(heatmap_flat, dim=-1)  # Softmax across H*W
        heatmap_norm = heatmap_norm.view(B, H, W)  # Reshape back to (B, H, W)
        
        # Create coordinate grids
        y_coords = torch.linspace(0, H - 1, H, device=heatmap.device).view(1, H, 1)  # Shape (1, H, 1)
        x_coords = torch.linspace(0, W - 1, W, device=heatmap.device).view(1, 1, W)  # Shape (1, 1, W)
        
        # Compute the weighted sum for each coordinate
        y_soft_argmax = torch.sum(heatmap_norm * y_coords, dim=(1, 2))  # Shape (B,)
        x_soft_argmax = torch.sum(heatmap_norm * x_coords, dim=(1, 2))  # Shape (B,)
        
        # Combine x and y coordinates
        soft_argmax_coords = torch.stack([y_soft_argmax, x_soft_argmax], dim=1)  # Shape (B, 2)
        return soft_argmax_coords

    def make_stop_mlp(self):
        in_dim = 24 * (2 ** self.num_deconv_layers)
        layers = []
        layers.append(nn.linear(in_dim, 32))
        layers.append(nn.ReLU(inplace=True))  
        layers.append(nn.linear(32, 2)) 
        layers.append(nn.Softmax()) 
        return nn.Sequential(*layers)
    
    def forward(self, x, seq_len):
        batch_size = x.shape[0]
        x = self.vit_extractor(x)
        x = x[:, 1:, :] # exclude the CLS token ## x: (Batch, patch_num, dim)
        
        x = x.permute(1, 0, 2) # x: (patch_num, batch, dim)

        x = x.unsqueeze(2) # x: (patch_num, batch, 1, dim)

        x = x.expand(-1, -1, seq_len, -1) # x: (patch_num, batch, seq_len, dim)

        p, b, s, d = x.shape
        ## memory heavy version
        #x = x.view(p * b, s, d)
        #x = self.lstm(x)
        #x = x.view(p, b, s, d)

        ## less memory
        lstm_out = []
        for i in range(len(p)):
            lstm_out.append(self.lstm(x[i]))
        x = torch.stack(lstm_out, dim = 0) # x: (patch_num, batch, seq_len, dim)

        x = x.permute(2, 1, 0, 3) # x: (seq_len, batch, patch_num, dim)

        heatmaps = []
        coords = []
        for i in range(len(s)):
            hid = self.vit2(x[i]) 
            hid = hid.reshape(batch_size, 24, 24, 768)
            hid = hid.permute(0, 3, 1, 2)
            hid = self.deconv_layrs(hid)
            hid = self.zero_conv(hid)
            heatmaps.append(hid)
            coord = self.soft_argmax_2d(hid)
            coords.append(coord)
        
        heatmaps = torch.stack(heatmaps, dim = 0)
        coords = torch.stack(coords, dim = 0) # coords (seq_len, batch_size, 2)
        coords = coords.permute(1, 0, 2)
        return heatmaps, coords
