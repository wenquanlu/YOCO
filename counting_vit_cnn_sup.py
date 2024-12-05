import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
import torch.nn.functional as F




class CountingViTCNNSup(nn.Module):

    def __init__(self, lstm_hidden_dim, num_deconv_layers=4):
        super(CountingViTCNNSup, self).__init__()
        self.vit_extractor = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden_dim, batch_first=True)
        self.num_deconv_layers = num_deconv_layers
        self.deconv_layrs = self.make_deconv_layer(num_deconv_layers, [256, 256, 128, 64])
        self.zero_conv = nn.Conv2d(64, 2, 1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 576, 768))
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(768)  # Batch normalization for convolutional layer
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(768)  # Batch normalization for convolutional layer
        self.conv3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(768)  # Batch normalization for convolutional layer
        self.conv4 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(768)  # Batch normalization for convolutional layer


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

    def hard_argmax_2d(self, heatmap):
        # Get the shape
        B, H, W = heatmap.shape
    
        # Flatten the heatmap and find the indices of the max values
        heatmap_flat = heatmap.view(B, -1)  # Flatten to (B, H*W)
        max_indices = torch.argmax(heatmap_flat, dim=-1)  # Shape (B,)
    
        # Convert the flat indices back to 2D coordinates
        y_coords = max_indices // W  # Integer division to get the row index
        x_coords = max_indices % W   # Modulus to get the column index
    
        # Combine x and y coordinates
        hard_argmax_coords = torch.stack([y_coords, x_coords], dim=1)  # Shape (B, 2)
        return hard_argmax_coords

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
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    print("nan!!!!!!!4")
        #for name, param in self.named_parameters():
        #    if torch.any(torch.isnan(param)):
        #        print(f"NaN found in {name}")
        #    if torch.any(torch.isinf(param)):
        #        print(f"Inf found in {name}")

        x = self.vit_extractor(x).last_hidden_state
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    print("nan!!!!!!!3")
        x = x[:, 1:, :] # exclude the CLS token ## x: (Batch, patch_num, dim)
        
        x = x.permute(1, 0, 2) # x: (patch_num, batch, dim)

        x = x.unsqueeze(2) # x: (patch_num, batch, 1, dim)

        x = x.expand(-1, -1, seq_len, -1) # x: (patch_num, batch, seq_len, dim)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    print("nan!!!!!!!2")
        p, b, s, d = x.shape
        ## memory heavy version
        #x = x.view(p * b, s, d)
        #x = self.lstm(x)
        #x = x.view(p, b, s, d)

        ## less memory
        lstm_out = []
        for i in range(p):
            lstm_output, _ = self.lstm(x[i])  # lstm_output: (seq_len, batch, hidden_dim)
            lstm_out.append(lstm_output)
        x = torch.stack(lstm_out, dim = 0) # x: (patch_num, batch, seq_len, dim)

        x = x.permute(2, 1, 0, 3) # x: (seq_len, batch, patch_num, dim)
        #if torch.isnan(x).any() or torch.isinf(x).any():
        #    print("nan!!!!!!!")
        heatmaps = []
        cum_heatmaps = []
        coords = []
        for i in range(s):
            hid = x[i].reshape(batch_size, 24, 24, 768)
            hid = hid.permute(0, 3, 1, 2)

            identity = hid
            hid = self.conv1(hid)
            hid = self.bn1(hid)
            hid = self.relu(hid)

            hid = self.conv2(hid)
            hid = self.bn2(hid)

            hid += identity
            hid = self.relu(hid)

            identity2 = hid

            hid = self.conv3(hid)
            hid = self.bn3(hid)
            hid = self.relu(hid)

            hid = self.conv4(hid)
            hid = self.bn4(hid)

            hid += identity2
            hid = self.relu(hid)

            hid = self.deconv_layrs(hid) 
            hid = self.zero_conv(hid) # (batch, 2, 384, 384)
            heatmap = hid[:,0,:,:].squeeze(1)
            cum_heatmap = hid[:,1,:,:].squeeze(1)
            heatmaps.append(heatmap)
            coord = self.hard_argmax_2d(heatmap)
            coords.append(coord)
            cum_heatmaps.append(cum_heatmap)

        heatmaps = torch.stack(heatmaps, dim = 0) #heatmaps (seq_len, batch, 384, 384)
        cum_heatmaps = torch.stack(cum_heatmaps, dim=0) #cum_heatmaps (seq_len, batch, 384, 384)

        coords = torch.stack(coords, dim = 0) # coords (seq_len, batch_size, 2)
        coords = coords.permute(1, 0, 2)
        heatmaps = heatmaps.permute(1, 0, 2, 3) # (batch, seq_len, H, W)
        cum_heatmaps = cum_heatmaps.permute(1, 0, 2, 3)
        return heatmaps, coords, cum_heatmaps

