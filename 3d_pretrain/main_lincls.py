import os
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from .train import train_function, test_function
from .model import MergeModel, MergeModel_Enet, MergeModel_ConvNeXt
from ..dataset import preprocess, CustomDataset

MODES = ["barlowtwins", "byol", "moco", "supcon", "simsiam", "tico", "unimoco", "vicreg", # SSL & SCL methods
         "unimoco_enet", "supcon_enet",                                                   # Robustness experiments (EfficientNetB0)
         "unimoco_convnext", "supcon_convnext",                                           # Robustness experiments (ConvNeXt-Tiny)
         "unimoco_ori",                                                                   # UniMoCo experiment
         "baseline", "baseline_enet", "baseline_convnext"]                                # Pure supervised learning with different backbone

parser = argparse.ArgumentParser(description='')

parser.add_argument("--epochs", type = int, default = 15,
                    help = "Number of training epochs")
parser.add_argument("--save_last_k", type = int, default = 2,
                    help = "Number of last epochs to save weights")
parser.add_argument("--batch_size", type = int, default = 4,
                    help = "Number of videos in a batch")
parser.add_argument("--accumulate", type = int, default = 4,
                    help = "Number of steps to gradient accumulation")
parser.add_argument("--num_workers", type = int, default = 0,
                    help = "Number of CPU workers")
parser.add_argument("--lr", type = float, default = 2e-4,
                    help = "Initial learning rate")
parser.add_argument("--wd", type = float, default = 0.01,
                    help = "Weight decay")
parser.add_argument("--warmup_ratio", type = float, default = 0.1,
                    help = "Ratio of warmup step to total training step. During the warmup step, the learning rate linearly increase from 0.0 to initial learning rate.")
parser.add_argument("--kfold", type = int, default = 5,
                    help = "Number of folds for stratified k-fold cross-validation")
parser.add_argument("--methods", nargs = '+', type = str, default = MODES, choices = MODES,
                    help = "List of models pretrained by different methods to run. Default: All methods")
parser.add_argument("--train_cnn", type = bool, default = True,
                    help = "Train 2D CNN or not. Default: True")
args = parser.parse_args()

DEBUG = True
TRAIN_CNN = args.train_cnn
DIR_NAME = os.path.dirname(__file__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Apply stratified k-fold cross-validation
    train, injury_folds, normal_folds = preprocess(n_splits = args.kfold)
    
    for k in range(args.kfold):
        for mode in args.methods:
            # Create a fold folder for saved weights and logs
            fold_path = os.path.join(DIR_NAME, f"weights/{mode}/fold{k+1}/")
            log_path = os.path.join(fold_path, 'log.txt')
            os.makedirs(fold_path, exist_ok = True)

            # Create Model
            if "enet" in mode: # EfficientNetB0
                model = MergeModel_Enet().to(DEVICE).float()
            elif "convnext" in mode: # ConvNeXt-Tiny
                model = MergeModel_ConvNeXt().to(DEVICE).float()
            else: # RegNetY-200MF
                model = MergeModel().to(DEVICE).float()
            
            # Load pretrained weights to model
            if "baseline" not in mode:
                pretrain_path = os.path.join(DIR_NAME, f"pretrain_weights/{mode}/fold{k+1}/weight.bin")
                pretrain_weights = torch.load(pretrain_path, weights_only = True)
                # Delete projector weights
                for key in list(pretrain_weights.keys()):
                    if 'projector' in key: del pretrain_weights[key]
                model.extractor.load_state_dict(pretrain_weights)

            # Freeze CNN weights (if necessary)
            for organ in ["full", "kidney", "liver", "spleen"]:
                extractor = getattr(model.extractor, f"{organ}_extractor")
                extractor.cnn.requires_grad_(TRAIN_CNN)

            # Load data
            train_df = pd.concat([injury_folds[k][0], normal_folds[k][0]]).reset_index(drop = True)
            val_df = pd.concat([injury_folds[k][1], normal_folds[k][1]]).reset_index(drop = True)

            train_dataset = CustomDataset(train_df[:128] if DEBUG else train_df, augmentation = True)
            val_dataset = CustomDataset(val_df, augmentation = False)
            
            train_loader = DataLoader(train_dataset,
                                      batch_size = args.batch_size,
                                      num_workers = args.num_workers,
                                      shuffle = True,
                                      drop_last = True)
            val_loader = DataLoader(val_dataset,
                                    batch_size = args.batch_size,
                                    num_workers = args.num_workers,
                                    shuffle = False,
                                    drop_last = False)
            
            ## Training settings
            # Optimizer
            optimizer = torch.optim.AdamW(params = model.parameters(), lr = 2e-4, weight_decay = 0.01)

            # Scheduler
            total_steps = int(len(train_loader) * args.epochs / args.accumulate)
            warmup_steps = int(total_steps * args.warmup_ratio)
            print('total_steps: ', total_steps)
            print('warmup_steps: ', warmup_steps)
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps = warmup_steps,
                                                        num_training_steps = total_steps)
            
            # Scaler
            scaler = torch.amp.GradScaler("cuda")
            
            print(f'{mode} training is starting...')
            for i in range(args.epochs):
                print(f'{i+1}th epoch training is starting...')
                # Training
                train_loss = train_function(model = model,
                                            optimizer = optimizer,
                                            scheduler = scheduler,
                                            scaler = scaler,
                                            loader = train_loader,
                                            iters_to_accumulate = args.accumulate)
                
                # Validation
                test_df, val_loss, message = test_function(model = model,
                                                           loader = val_loader,
                                                           input_df = val_loader.dataset.df.copy())

                # Save model weights
                model_save_path = os.path.join(fold_path, f"weight_test.bin")
                # model_save_path = os.path.join(fold_path, f'epoch{i+1}_valloss{round(val_loss.item(), 4)}.bin')
                if i >= (args.epochs - args.save_last_k):
                    torch.save({"extractor": model.extractor.state_dict(), 
                                "classifier": model.classifier.state_dict()}, model_save_path)
                
                # Save validation history
                with open(log_path, 'a+') as logger:
                    logger.write(f'{message}\n')
