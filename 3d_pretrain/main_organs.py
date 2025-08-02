import os
import torch
import pandas as pd
import numpy as np
import argparse
import yaml
from torch.utils.data import DataLoader
from .train import pretrain_function
from .ssl_methods import *
from ..dataset import preprocess, CustomDataset

MODES = ["barlowtwins", "byol", "moco", "supcon", "simsiam", "tico", "unimoco", "vicreg", # SSL & SCL methods
         "unimoco_enet", "supcon_enet",                                                   # Robustness experiments (EfficientNetB0)
         "unimoco_convnext", "supcon_convnext",                                           # Robustness experiments (ConvNeXt-Tiny)
         "unimoco_ori",                                                                   # UniMoCo experiment
         "baseline", "baseline_enet", "baseline_convnext"]                                # Baseline with different backbone

parser = argparse.ArgumentParser(description='')

parser.add_argument("--epochs", type = int, default = 10,
                    help = "Number of training epochs")
parser.add_argument("--batch_size", type = int, default = 5,
                    help = "Number of videos in a batch")
parser.add_argument("--accumulate", type = int, default = 16,
                    help = "Number of steps to gradient accumulation")
parser.add_argument("--num_workers", type = int, default = 0,
                    help = "Number of CPU workers")
parser.add_argument("--lr", type = float, default = 2e-4,
                    help = "Initial learning rate")
parser.add_argument("--wd", type = float, default = 1e-4,
                    help = "Weight decay")
parser.add_argument("--warmup_ratio", type = float, default = 2/10, # 2 warmup epochs out of 10 total epochs
                    help = "Ratio of warmup step to total training step. During the warmup step, the learning rate linearly increase from 0.0 to initial learning rate.")
parser.add_argument("--kfold", type = int, default = 5,
                    help = "Number of folds for stratified k-fold cross-validation")
parser.add_argument("--methods", nargs = '+', type = str, default = MODES, choices = MODES,
                    help = "List of models pretrained by different methods to run. Default: All methods")
args = parser.parse_args()

DEBUG = True
DIR_NAME = os.path.dirname(__file__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(DIR_NAME, "hparam.yaml")) as stream:
    INIT_M = yaml.safe_load(stream)["tico"]["init_m"] # Load TiCo initial momentum from YAML file
TICO_M = INIT_M + (1 - INIT_M) * np.sin((np.pi/2) * np.arange(args.epochs) / (args.epochs - 1)) # Define TiCo's momentum for each epoch

MODEL_DICT = {"barlowtwins": BarlowTwins, "byol": BYOL, "moco": MoCo, "supcon": SupCon, 
              "unimoco": UniMoCo, "tico": TiCo, "vicreg": VICReg, "simsiam": SimSiam,
              "supcon_enet": SupCon_Enet, "unimoco_enet": UniMoCo_Enet,
              "supcon_convnext": SupCon_ConvNeXt, "unimoco_convnext": UniMoCo_ConvNeXt, 
              "unimoco_ori": UniMoCo_Ori}

if __name__ == "__main__":
    # Apply stratified k-fold cross-validation
    _, injury_folds, normal_folds = preprocess(n_splits = args.kfold)
    for k in range(args.kfold):
        for mode in args.methods:
            # Create a fold folder for saved weights
            fold_path = os.path.join(DIR_NAME, f"pretrain_weights/{mode}/fold{k+1}/")
            os.makedirs(fold_path, exist_ok = True)

            # Load data
            pretrain_df = pd.concat([injury_folds[k][0], normal_folds[k][0]]).reset_index(drop = True)

            pretrain_dataset = CustomDataset(pretrain_df[:128] if DEBUG else pretrain_df)
            
            pretrain_loader = DataLoader(pretrain_dataset,
                                         batch_size = args.batch_size,
                                         num_workers = args.num_workers,
                                         shuffle = True,
                                         drop_last = True)

            # Create model
            model = MODEL_DICT[mode]().to(DEVICE)
            
            # Optimizer
            optimizer = torch.optim.AdamW(params = model.parameters(), lr = args.lr, weight_decay = args.wd)
            
            # Scheduler
            total_steps = int(len(pretrain_loader) * args.epochs / args.accumulate)
            warmup_steps = int(total_steps * args.warmup_ratio)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                                 start_factor = 0.01, end_factor = 1.0, 
                                                                 total_iters = warmup_steps)
            cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max = (total_steps - warmup_steps) // 1, # 0.5 waves
                                                                       eta_min = args.lr * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                              schedulers = [warmup_scheduler, cos_scheduler],
                                                              milestones = [warmup_steps])

            # Scaler
            scaler = torch.amp.GradScaler("cuda")
            
            print(f'{mode} pretraining is starting...')
            for curr_epoch in range(args.epochs):
                train_loss, history = pretrain_function(model = model,
                                                        optimizer = optimizer,
                                                        scheduler = scheduler,
                                                        scaler = scaler,
                                                        loader = pretrain_loader,
                                                        iters_to_accumulate = args.accumulate,
                                                        tico_m = TICO_M[curr_epoch])

            # Save pretrained model weights
            torch.save(model.online_encoder.state_dict(),
                       os.path.join(fold_path, 'weight_test.bin' if DEBUG else 'weight.bin'))