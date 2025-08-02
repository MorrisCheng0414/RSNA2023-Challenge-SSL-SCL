import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob
from torch.utils.data import DataLoader
from dataset import preprocess, CustomDataset
from model import Custom3DCNN
from sklearn.manifold import TSNE
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def output_embeddings(extractor,
                      loader):
    extractor.train()
    
    embeds = {"full": [],
              "kidney": [],
              "liver": [],
              "spleen": []}
    labels = {"full": [],
              "kidney": [],
              "liver": [],
              "spleen": []}
    embeds = [[] for _ in range(4)]
    labels = [[] for _ in range(4)]
    
    for sample in tqdm(loader):
        images, crop_kidney, crop_liver, crop_spleen, _, bowel, extravasation, kidney, liver, spleen, *_ = sample
        X = torch.stack([images, crop_kidney, crop_liver, crop_spleen]).to(device) # images: (4, batch_size, img_num, 256, 256)
        # X = X.permute(1, 0, 2, 3, 4) # X: (4, batch_size, img_num, 256, 256)

        with torch.no_grad():
            with torch.amp.autocast(device_type = "cuda", dtype = torch.float16):
                embed = extractor(*X).detach().cpu().numpy()
                full_embed, kidney_embed, liver_embed, spleen_embed = embed 
                
        embeds[0].extend(full_embed)
        embeds[1].extend(kidney_embed)
        embeds[2].extend(liver_embed)
        embeds[3].extend(spleen_embed)

        labels[0].extend(bowel | extravasation)
        labels[1].extend(kidney)
        labels[2].extend(liver)
        labels[3].extend(spleen)
                
    embeds = [np.asarray(embed) for embed in embeds]
    labels = [np.asarray(label) for label in labels]

    return ({"full": embeds[0], "kidney": embeds[1], "liver": embeds[2], "spleen": embeds[3]},
            {"full": labels[0], "kidney": labels[1], "liver": labels[2], "spleen": labels[3]})

def get_tsne(model, loader, folder_path):
    embeds, labels = output_embeddings(model, loader)
    fig, axs = plt.subplots(1, 4, figsize = (16, 4))

    label_bi = ["Healthy", "Injury"]
    label_mul = ["Healthy", "Low", "High"]
    colors = ["royalblue", "darkorange", "red"]

    for idx, organ_name in enumerate(["full", "kidney", "liver", "spleen"]):
        tsne = TSNE(n_components = 2, 
                    perplexity = 60,
                    learning_rate = "auto", 
                    metric = "cosine", # The loss is computes by cosine similarity
                    init = "pca",
                    random_state = 42).fit_transform(embeds[organ_name])

        label_unqiue = np.unique(labels[organ_name])
        label_name = label_bi if label_unqiue.shape[0] == 2 else label_mul
        
        for label in label_unqiue:
            tsne_idx = np.where(labels[organ_name] == label)
            axs[idx].scatter(tsne[tsne_idx, 0], tsne[tsne_idx, 1], 
                            color = colors[label],
                            s = 10, 
                            label = label_name[label])
        axs[idx].legend()
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        axs[idx].set_title(f"T-SNE for {organ_name}")

    fig.tight_layout()
    fig.savefig(os.path.join(folder_path, "tsne_svg.svg"))
    fig.savefig(os.path.join(folder_path, "tsne_png.png"))

def get_loss_curve(loss_history, folder_path):
    fig, axs = plt.subplots(1, 1, figsize = (10, 5))
    class_names = ["bowel", "extravasation", "kidney", "liver", "spleen"]
    for organ_name, loss in zip(class_names, loss_history):
        axs.plot(loss, label = organ_name)
    axs.xaxis.get_major_locator().set_params(integer = True)
    axs.set_xlabel("steps")
    axs.set_ylabel("loss")
    axs.set_title("Training loss")
    axs.legend(loc = "best")

    fig.tight_layout()
    fig.savefig(os.path.join(folder_path, "loss_svg.svg"))
    fig.savefig(os.path.join(folder_path, "loss_png.png"))

if __name__ == "__main__":
    methods = ["barlowtwins", "byol", "moco", "simsiam", "supcon", "supcon_enet", "tico", "unimoco", "unimoco_enet", "vicreg"]
    # methods = ["regnety"]

    k_fold = 5
    batch_size = 4
    num_workers = 0

    train, injury_folds, normal_folds = preprocess(n_splits = k_fold)
    for method in methods:
        for fold in range(1):
            # weight_path = "/kaggle/working/12th_regnety_reproduce/weights/fold1/epoch4_valloss0.4928.bin"
            weight_path = glob(f"/kaggle/working/RSNA_23_3D_SSL/weights/{method}/fold{fold + 1}/epoch15*.bin")[-1]
            print(weight_path)
            # weight = torch.load(weight_path, weights_only = True)
            weight = torch.load(weight_path, weights_only = True)["extractor"]
            
            model = Custom3DCNN().to(device).float()
            # model.load_state_dict(weight, strict = False)
            model.load_state_dict(weight)

             # Load data
            train_df = pd.concat([injury_folds[fold][0], normal_folds[fold][0]]).reset_index(drop = True)
            # val_df = pd.concat([injury_folds[fold][1], normal_folds[fold][1]]).reset_index(drop = True)

            train_dataset = CustomDataset(train_df, augmentation = True)
            # val_dataset = CustomDataset(val_df, augmentation = False)
            
            train_loader = DataLoader(train_dataset,
                                    batch_size = batch_size,
                                    num_workers = num_workers,
                                    shuffle = True,
                                    drop_last = True)
            # val_loader = DataLoader(val_dataset,
            #                         batch_size = batch_size,
            #                         num_workers = num_workers,
            #                         shuffle = False,
            #                         drop_last = False)
            
            folder_path = os.path.join("/kaggle/working/RSNA_23_3D_SSL/tsne", method, f"fold{fold+1}")
            os.makedirs(folder_path, exist_ok = True)
            get_tsne(model, train_loader, folder_path)
            

            

