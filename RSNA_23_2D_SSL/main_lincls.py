import os
import torch
from glob import glob
import pandas as pd
from torch.utils.data import DataLoader
from train import train_function, test_function
from dataset import preprocess, CustomDataset
from transformers import get_cosine_schedule_with_warmup
from model import MergeModel
from tsne import get_loss_curve
from utils import clear_folds_weights

DEBUG = False
TRAIN_CNN = True # set True to train cnn weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 3 if DEBUG else 20 # default: 15 -> 10
early_stop_epoch = 15
batch_size = 4 # default: 5 -> 6
num_workers = 0
iters_to_accumulate = 4 # Defualt: 4 -> 16
warmup_ratio = 0.1

modes = ["barlowtwins", "byol", "moco", "simsiam", "tico", "vicreg"]

k_fold = 5

if __name__ == "__main__":
    # Clear weights
    # clear_folds_weights("/kaggle/working/RSNA_23_2D_SSL/weights")

    train, injury_folds, normal_folds = preprocess(n_splits = k_fold)
    for k in range(k_fold):
        for mode in modes:
            # Create a fold folder for saved weights and logs
            fold_path = f"/kaggle/working/RSNA_23_2D_SSL/weights/{mode}/fold{k+1}/"
            os.makedirs(fold_path, exist_ok = True)

            model_save_path = os.path.join(fold_path, 'weight.bin')

            ## Load pretrain weights & initialize model
            pretrain_weights = torch.load(f"/kaggle/working/RSNA_23_2D_SSL/pretrain_weights/{mode}/fold{k+1}/weight.bin", weights_only = True)        
            for key in list(pretrain_weights.keys()):
                if 'projector' in key: del pretrain_weights[key]

            model = MergeModel().to(device).float()

            organs = ["full", "kidney", "liver", "spleen"]
            for organ in organs:
                extractor = getattr(model.extractor, f"{organ}_extractor")
                extractor.cnn.load_state_dict(pretrain_weights[f"{organ}_cnn"]) # delete "state_dict" later
                extractor.cnn.requires_grad_(TRAIN_CNN)

            # Load data
            train_df = pd.concat([injury_folds[k][0], normal_folds[k][0]]).reset_index(drop = True)
            val_df = pd.concat([injury_folds[k][1], normal_folds[k][1]]).reset_index(drop = True)

            train_dataset = CustomDataset(train_df, augmentation = True)
            val_dataset = CustomDataset(val_df, augmentation = False)
            
            train_loader = DataLoader(train_dataset,
                                    batch_size = batch_size,
                                    num_workers = num_workers,
                                    shuffle = True,
                                    drop_last = True)
            val_loader = DataLoader(val_dataset,
                                    batch_size = batch_size,
                                    num_workers = num_workers,
                                    shuffle = False,
                                    drop_last = False)
            
            # Training settings
            train_history = [[] for _ in range(5)]
            optimizer = torch.optim.AdamW(params = model.parameters(), lr = 2e-4, weight_decay = 0.01)

            total_steps = int(len(train_loader) * epoch / iters_to_accumulate)
            warmup_steps = int(total_steps * warmup_ratio)
            print('total_steps: ', total_steps)
            print('warmup_steps: ', warmup_steps)

            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps = warmup_steps,
                                                        num_training_steps = total_steps)
            scaler = torch.amp.GradScaler("cuda")
            print(f'{mode} training is starting...')
            for i in range(epoch):
                print(f'{i+1}th epoch training is start...')
                if i == early_stop_epoch:
                    break

                train_loss = train_function(model,
                                            optimizer,
                                            scheduler,
                                            scaler,
                                            train_loader,
                                            iters_to_accumulate)

                train_history = [train_history[idx] + train_loss[idx] for idx in range(len(train_history))]

                _, val_loss, message = test_function(model,
                                                    val_loader,
                                                    val_loader.dataset.df.copy())

                model_save_path = os.path.join(fold_path, f'epoch{i+1}_valloss{round(val_loss.item(), 4)}.bin')
                if i >= epoch - 2:
                    torch.save({"extractor": model.extractor.state_dict(), 
                                "classifier": model.classifier.state_dict()}, model_save_path)
                # Save train history
                log_path = os.path.join(fold_path, 'log.txt')
                # message['log'] = f"epoch : {i+1}, 'valloss' : {val_loss.item() :.4f}"
                with open(log_path, 'a+') as logger:
                    logger.write(f'{message}\n')

            # os.rename(model_save_path, os.path.join(fold_path, f'epoch{best_epoch}_valloss{round(best_loss.item(), 4)}.bin'))
            get_loss_curve(train_history, fold_path)