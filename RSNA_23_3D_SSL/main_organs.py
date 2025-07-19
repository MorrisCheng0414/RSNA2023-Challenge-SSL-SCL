import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from train import pretrain_function
from dataset import preprocess, CustomDataset
from ssl_methods import *
from utils import clear_folds_weights
from tsne import get_tsne, get_loss_curve

DEBUG = False

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 3 if DEBUG else 10 # default: 20 -> 10
    num_workers = 0
    batch_size = 5 # default: 5 -> 6
    iters_to_accumulate = 16 # Defualt: 16 -> 32

    init_m = 0.99 # default: 0.5 -> 0.9
    tico_m = init_m + (1 - init_m) * np.sin((np.pi/2) * np.arange(epoch) / (epoch - 1))

    model_map = {"barlowtwins": BarlowTwins, "byol": BYOL, "moco": MoCo, "supcon": SupCon, 
                 "unimoco": UniMoCo, "tico": TiCo, "vicreg": VICReg, "simsiam": SimSiam,
                 "supcon_enet": SupCon_Enet, "unimoco_enet": UniMoCo_Enet,
                 "supcon_convnext": SupCon_ConvNeXt, "unimoco_convnext": UniMoCo_ConvNeXt, "unimoco_ori": UniMoCo_Ori}
    modes = list(model_map.keys())
    modes = ["unimoco_ori"] # for debug, delete later
    
    for mode in modes:
        if mode not in model_map: 
            raise ValueError(f"Invalid mode: {mode}")

    # # Clear weights
    # clear_folds_weights("/kaggle/working/RSNA_23_3D_SSL/pretrain_weights")
    
    # Load data
    k_fold = 5
    _, injury_folds, normal_folds = preprocess(n_splits = k_fold)
    for k in range(k_fold):
        # if k != 0: break # delete later

        for mode in modes:
            fold_path = f"/kaggle/working/RSNA_23_3D_SSL/pretrain_weights/{mode}/fold{k+1}/"
            os.makedirs(fold_path, exist_ok = True)
            pretrain_df = pd.concat([injury_folds[k][0], normal_folds[k][0]]).reset_index(drop = True)

            pretrain_dataset = CustomDataset(pretrain_df[:128] if DEBUG else pretrain_df)
            
            pretrain_loader = DataLoader(pretrain_dataset,
                                         batch_size = batch_size,
                                         num_workers = num_workers,
                                         shuffle = True,
                                         drop_last = True)

            # Create model
            model = model_map[mode]().to(device)
            
            ## MoCo pretrain extractor
            lr = 2e-4 # default: 2e-4 -> 5e-5
            warmup_epoch = 2 # default: 5 -> 2
            warmup_steps = len(pretrain_loader) * warmup_epoch // iters_to_accumulate

            optimizer = torch.optim.AdamW(params = model.parameters(), lr = lr, weight_decay = 1e-4)
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                                    start_factor = 0.01, end_factor = 1.0, 
                                                                    total_iters = warmup_steps)
            cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                        T_max = len(pretrain_loader) * (epoch - warmup_epoch) // iters_to_accumulate // 1, # 0.5 waves
                                                                        eta_min = lr * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, 
                                                                schedulers = [warmup_scheduler, cos_scheduler],
                                                                milestones = [warmup_steps])

            scaler = torch.amp.GradScaler("cuda")
            loss_history = [[], [], [], []]
            
            print(f'{mode} pretraining is starting...')
            for curr_epoch in range(epoch):
                train_loss, history = pretrain_function(model = model,
                                                        optimizer = optimizer,
                                                        scheduler = scheduler,
                                                        scaler = scaler,
                                                        loader = pretrain_loader,
                                                        iters_to_accumulate = iters_to_accumulate,
                                                        tico_m = tico_m[curr_epoch])
                for idx, loss in enumerate(history):
                    loss_history[idx].extend(loss)

            torch.save(model.online_encoder.state_dict(),
                       os.path.join(fold_path, 'weight.bin'))
            
            # model.online_encoder.load_state_dict(torch.load(os.path.join(fold_path, 'weight.bin')))
            model.online_encoder.full_projector = torch.nn.Identity()
            model.online_encoder.kidney_projector = torch.nn.Identity()
            model.online_encoder.liver_projector = torch.nn.Identity()
            model.online_encoder.spleen_projector = torch.nn.Identity()
            get_tsne(model.online_encoder, pretrain_loader, fold_path)
            get_loss_curve(loss_history, fold_path)