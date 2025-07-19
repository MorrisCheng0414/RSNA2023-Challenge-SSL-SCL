import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Custom2DCNN
from dataset import preprocess, CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def aggregate_embeds(frame_embeds, video_label):
    video_embed = None
    
    if video_label == 0: # healthy
        video_embed = np.mean(frame_embeds, axis = 0)
        
    elif video_label != 0: # injury or low or high
        kmeans = KMeans(n_clusters = 2, random_state = 42).fit(frame_embeds)
        labels = kmeans.labels_
        cluster_counts = np.bincount(labels)
        injury_cluster_idx = np.argmin(cluster_counts)
        video_embed = kmeans.cluster_centers_[injury_cluster_idx]
        
    return video_embed

def output_embeddings(model, loader):
    model.eval()
    
    embeds = {"full": [], "kidney": [], "liver": [], "spleen": []}
    labels = {"full": [], "kidney": [], "liver": [], "spleen": []}
    embeds = [[] for _ in range(4)]
    labels = [[] for _ in range(4)]
    
    for step in tqdm(loader):
        images, crop_kidney, crop_liver, crop_spleen, _, bowel, extravasation, kidney, liver, spleen, _, _ = step
        X = torch.stack([images, crop_kidney, crop_liver, crop_spleen]).to(device) # images: (4, batch_size, img_num, 256, 256)

        with torch.no_grad():
            with torch.amp.autocast(device_type = "cuda"):
                full_embed, kidney_embed, liver_embed, spleen_embed = model(*X).detach().cpu().numpy() # model_output: (4, batch_size, 32, embed_size)

                embeds[0].extend(full_embed)
                embeds[1].extend(kidney_embed)
                embeds[2].extend(liver_embed)
                embeds[3].extend(spleen_embed)

                labels[0].extend(bowel.numpy() + extravasation.numpy())
                labels[1].extend(kidney.numpy())
                labels[2].extend(liver.numpy())
                labels[3].extend(spleen.numpy())
                
    embeds = [np.asarray(embed) for embed in embeds] # embeds: (4, loader_len, 32, 368)
    labels = [np.asarray(label) for label in labels] # embeds: (4, loader_len)

    # return embeds, labels
    for idx, (embed, label) in enumerate(zip(embeds, labels)):
        embeds[idx] = np.asarray([aggregate_embeds(e, l) for e, l in zip(embed, label)])
    
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
                    learning_rate = 'auto', 
                    init = 'pca', 
                    perplexity = 10).fit_transform(embeds[organ_name])

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
    # TODO: complete t-SNE plotting function
    k_fold = 5
    _, injury_folds, normal_folds = preprocess(n_splits = k_fold)
    fold_path = "/kaggle/working/RSNA_23_2D_SSL/pretrain_weights/fold1/"
    os.makedirs(fold_path, exist_ok = True)
    pretrain_df = pd.concat([injury_folds[0][0], normal_folds[0][0]]).reset_index(drop = True)

    pretrain_dataset = CustomDataset(pretrain_df[:128])
    
    pretrain_loader = DataLoader(pretrain_dataset,
                                    batch_size = 3,
                                    num_workers = 0,
                                    shuffle = True,
                                    drop_last = True)
    
    model = Custom2DCNN().to(device)
    get_tsne(model, pretrain_loader, fold_path)