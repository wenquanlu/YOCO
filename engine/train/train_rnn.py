import os
import wandb
import random
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.counting_rnn import CountingRNN
from data.dataset import WiderFaceDataset
from data.parse_dataset_yoco3k import parse_dataset_yoco3k
from utils.utils import WarmUpLR, add_gaussians_to_heatmaps_batch, sort_naive, save_checkpoint

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, iteration, wandb):
    for _, imgs, coords, seq_lens, max_seq_len, _ in tqdm(dataloader):
        # coords: [batch, max_seq_len, 2]
        # max_seq_len = max(seq_lens) + 1
        imgs = imgs.cuda()
        coords = coords.cuda()
        seq_lens = seq_lens.cuda()
        max_seq_len = max_seq_len.cuda()

        ## TODO
        predicted_heatmaps, predicted_coords = model(imgs, max_seq_len)
        #coords = match_coords(predicted_coords, coords)
        coords = sort_naive(coords)
        heatmaps = add_gaussians_to_heatmaps_batch(predicted_heatmaps, coords)
        loss = model.compute_loss(predicted_heatmaps, heatmaps, predicted_coords, seq_lens, max_seq_len, alpha=0.1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        wandb.log({"epoch": epoch, "iteration": iteration, "train_loss": loss, "LR": optimizer.param_groups[0]['lr']})
        iteration += 1

    return model, optimizer, iteration


def train_rnn(args):
    config = OmegaConf.load(args.config_file)
    wandb.init(
        project="yoco3k-12",
        config={}
    )

    data, labels = parse_dataset_yoco3k(args.train_set)
    train_dataset = WiderFaceDataset(data, labels)
    dataloader = DataLoader(train_dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True, 
                            num_workers=4, 
                            collate_fn=train_dataset.custom_collate_fn)

    model = CountingRNN(768)
    if args.state_dict != "":
        print("load", args.state_dict)
        model.load_state_dict(torch.load(args.state_dict)["model_state_dict"])
    model.cuda()

    optimizer = Adam(model.parameters(), lr=1e-6)
    scheduler_config = config.training.scheduler
    warmup_epochs = scheduler_config["warmup_epochs"]
    warmup_lr = scheduler_config["warmup_lr"]
    base_lr = scheduler_config["base_lr"]  
    iteration_per_epoch = len(dataloader)
    warmup_scheduler = WarmUpLR(optimizer, warmup_epochs * iteration_per_epoch, warmup_lr, base_lr)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    iteration = 0

    for epoch in range(config.training.epochs):
        model.train()
        if epoch == warmup_epochs:
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max = config.training.epochs * iteration_per_epoch, 
                eta_min=scheduler_config["eta_min"],
            )
        
        model, optimizer, iteration = train_one_epoch(model, dataloader, optimizer, warmup_scheduler, epoch, iteration, wandb)
        save_checkpoint(model, optimizer, cosine_scheduler, epoch, warmup_epochs, args.save_dir)