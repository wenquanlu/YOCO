import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTModel


class CountingRNNSup(nn.Module):

    def __init__(self, lstm_hidden_dim, num_deconv_layers=4):
        super(CountingRNNSup, self).__init__()
        self.num_deconv_layers = num_deconv_layers

        self.vit_extractor = ViTModel.from_pretrained("google/vit-base-patch16-384")
        self.lstm = nn.LSTM(input_size=768, hidden_size=lstm_hidden_dim, batch_first=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, 576, 768))
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(768)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(768)
        self.conv3 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(768)
        self.conv4 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(768)
        self.deconv_layrs = self.make_deconv_layer(in_channel=768, num_layers=num_deconv_layers, num_filters=[256, 256, 128, 64])
        self.zero_conv = nn.Conv2d(64, 2, 1)

    def make_deconv_layer(self, in_channel, num_layers, num_filters):
        layers = []
        for i in range(num_layers):
            filters = num_filters[i]
            deconv_layer = nn.ConvTranspose2d(
                in_channels=in_channel,
                out_channels=num_filters[i],
                kernel_size=4,
                stride=2,
                padding=1
            )     
            layers.append(deconv_layer)
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.ReLU(inplace=True))   
            in_channel = num_filters[i]

        return nn.Sequential(*layers)

    def hard_argmax_2d(self, heatmap):
        B, H, W = heatmap.shape
    
        heatmap_flat = heatmap.view(B, -1)  # shape (B, H*W)
        max_indices = torch.argmax(heatmap_flat, dim=-1)  # Shape (B,)
    
        y_coords = max_indices // W
        x_coords = max_indices % W
    
        hard_argmax_coords = torch.stack([y_coords, x_coords], dim=1)  # Shape (B, 2)

        return hard_argmax_coords
    
    def forward(self, x, seq_len):
        x = self.vit_extractor(x).last_hidden_state

        x = x[:, 1:, :] # x: (Batch, patch_num, dim)
        x = x.permute(1, 0, 2) # x: (patch_num, batch, dim)
        x = x.unsqueeze(2) # x: (patch_num, batch, 1, dim)
        x = x.expand(-1, -1, seq_len, -1) # x: (patch_num, batch, seq_len, dim)
        p, b, s, d = x.shape

        lstm_out = []
        for i in range(p):
            lstm_output, _ = self.lstm(x[i])  # lstm_output: (seq_len, batch, hidden_dim)
            lstm_out.append(lstm_output)
        x = torch.stack(lstm_out, dim = 0) # x: (patch_num, batch, seq_len, dim)
        x = x.permute(2, 1, 0, 3) # x: (seq_len, batch, patch_num, dim)

        heatmaps = []
        cum_heatmaps = []
        coords = []
        for i in range(s):
            hid = x[i].reshape(b, 24, 24, 768)
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

    def compute_loss(self, predicted_heatmaps, heatmaps, predicted_coords, predicted_cum_heatmaps, cum_heatmaps, seq_lens, max_seq_len, alpha=0.1):
        """
        heatmaps (batch, max_seq_len, H, W)
        predicted_heatmap (batch, max_seq_len, H, W)
        predicted_coords (batch, max_seq_len, 2)
        coords (batch, max_seq_len, 2)
        """
        seq_indices = torch.arange(max_seq_len, device=seq_lens.device)

        heatmap_mask = (seq_indices.unsqueeze(0) <= seq_lens.unsqueeze(1)).int()
        heatmap_mask = heatmap_mask.unsqueeze(-1).unsqueeze(-1)

        mse_loss = F.mse_loss(heatmaps, predicted_heatmaps, reduction='none')
        masked_mse = mse_loss * heatmap_mask
        cum_loss = F.mse_loss(cum_heatmaps, predicted_cum_heatmaps, reduction='none')
        masked_cum_mse = cum_loss * heatmap_mask
        l2_loss = (masked_mse.sum()+masked_cum_mse.sum()) / (2 * heatmap_mask.sum() * 384 * 384)

        return l2_loss 

