import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as v2
import yaml
from PIL import Image
from glob import glob
from sklearn.model_selection import KFold

IMG_SIZE = 256
DIR_NAME = os.path.dirname(__file__)
with open(os.path.join(DIR_NAME, "dataset_dir.yaml")) as stream:
    DATASET_DIR = yaml.safe_load(stream)["dataset_dir"]

def preprocess(n_splits = 5):
    train_series_meta = pd.read_csv(os.path.join(DIR_NAME, "data/train_series_meta.csv"))
    train_series_meta = train_series_meta.sort_values(by='patient_id').reset_index(drop=True)
    train = pd.read_csv(os.path.join(DIR_NAME, "data/train.csv"))
    train = train.sort_values(by='patient_id').reset_index(drop=True)
    _train = []
    series_ids = []
    for i in range(len(train_series_meta)):
        patient_id, series_id, _, _ = train_series_meta.loc[i]
        sample = train[train['patient_id']==patient_id]
        _train.append(sample)
        series_ids.append(int(series_id))

    _train = pd.concat(_train).reset_index(drop=True)
    _train['series_id'] = series_ids

    train = _train

    injury_train = train[train['any_injury']==1].reset_index(drop=True)
    normal_train = train[train['any_injury']==0].reset_index(drop=True)

    kf = KFold(n_splits=n_splits)
    injury_folds = []
    for i, (train_index, test_index) in enumerate(kf.split(injury_train)):
        train_df = injury_train.loc[train_index]
        val_df = injury_train.loc[test_index]
        injury_folds.append([train_df, val_df])


    kf = KFold(n_splits=n_splits)
    normal_folds = []
    for i, (train_index, test_index) in enumerate(kf.split(normal_train)):
        train_df = normal_train.loc[train_index]
        val_df = normal_train.loc[test_index]
        normal_folds.append([train_df, val_df])
    return train, injury_folds, normal_folds

class LinclsAug(nn.Module):
    def __init__(self, prob = 0.5, s = IMG_SIZE):
        super(LinclsAug, self).__init__()
        self.prob = prob

        self.do_random_rotate = v2.RandomRotation(
            degrees = (-45, 45),
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            expand = False,
            center = None,
            fill = 0
        )
        self.do_random_scale = v2.ScaleJitter(
            target_size = [s, s],
            scale_range = (0.8, 1.2),
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            antialias = True)

        self.do_random_crop = v2.RandomCrop(
            size = [s, s],
            #padding = None,
            pad_if_needed = True,
            fill = 0,
            padding_mode = 'constant'
        )

        self.do_horizontal_flip = v2.RandomHorizontalFlip(self.prob)
        self.do_vertical_flip = v2.RandomVerticalFlip(self.prob)

    def forward(self, x):
        if np.random.rand() < self.prob:
            x = self.do_random_rotate(x)

        if np.random.rand() < self.prob:
            x = self.do_random_scale(x)
            x = self.do_random_crop(x)

        x = self.do_horizontal_flip(x)
        x = self.do_vertical_flip(x)
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, augmentation=False):
        self.df = df
        self.label_columns = self.df.columns[1:-2]

        self.img_size = IMG_SIZE

        self.augmentation = augmentation
        self.aug_function = LinclsAug()

        self.sample_weights = {
            'bowel' : {0:1, 1:2},
            'extravasation' : {0:1, 1:6},
            'kidney' : {0:1, 1:2, 2:4},
            'liver' : {0:1, 1:2, 2:4},
            'spleen' : {0:1, 1:2, 2:4},
            'any_injury' : {0:1, 1:6}
            }

        self.sample_weights = {
            'bowel' : {0:1, 1:1},
            'extravasation' : {0:1, 1:1},
            'kidney' : {0:1, 1:1, 2:1},
            'liver' : {0:1, 1:1, 2:1},
            'spleen' : {0:1, 1:1, 2:1},
            'any_injury' : {0:1, 1:1}
            }

    def __len__(self):
        return len(self.df)

    def load_sample_png(self, png_path, default_tensor):
        if not os.path.exists(png_path):
            return default_tensor
        
        # Retrieve & sort the PNGs
        png_list = sorted(glob(png_path + "/*"), key = lambda x: int(x.rsplit("/")[-1][ : -4]))
        # Load the image by index
        png_list = np.array([np.array(Image.open(png).resize((IMG_SIZE, IMG_SIZE)).convert("L")) for png in png_list])
        # Resample the images to 96
        sample_idx = np.linspace(start = 0, stop = len(png_list), num = 96, endpoint = False, dtype = np.uint8)
        png_list = np.array([png_list[idx] for idx in sample_idx])
        
        return torch.tensor(png_list, dtype = torch.float32) # size: (96, 256, 256)

    def __getitem__(self, index):
        sample = self.df.loc[index]
        patient_id, series_id, any_injury = int(sample['patient_id']), int(sample['series_id']), int(sample['any_injury'])
        patient_series_path = os.path.join(DATASET_DIR, f"{patient_id}", f"{series_id}")
        
        images_path = os.path.join(patient_series_path, "images")
        crop_kidney_path = os.path.join(patient_series_path, "kidney")
        crop_liver_path = os.path.join(patient_series_path, "liver")
        crop_spleen_path = os.path.join(patient_series_path, "spleen")
        
        # Load and resample the numpy array to fixed number of frames. Output as float tensor
        images = self.load_sample_png(images_path, None)
        crop_kidney = self.load_sample_png(crop_kidney_path, images)
        crop_liver = self.load_sample_png(crop_liver_path, images)
        crop_spleen = self.load_sample_png(crop_spleen_path, images)

        # Apply image augmentation if specified
        if self.augmentation:
            images = self.aug_function(images)
            crop_kidney = self.aug_function(crop_kidney)
            crop_liver = self.aug_function(crop_liver)
            crop_spleen = self.aug_function(crop_spleen)

        label = torch.tensor(sample[self.label_columns].values, dtype=torch.long)

        bowel = label[0:2].argmax()
        extravasation = label[2:4].argmax()
        kidney = label[4:7].argmax()
        liver = label[7:10].argmax()
        spleen = label[10:13].argmax()
        any_injury = torch.tensor(any_injury, dtype=torch.float)

        sample_weights = torch.tensor([
            self.sample_weights['bowel'][bowel.tolist()],
            self.sample_weights['extravasation'][extravasation.tolist()],
            self.sample_weights['kidney'][kidney.tolist()],
            self.sample_weights['liver'][liver.tolist()],
            self.sample_weights['spleen'][spleen.tolist()],
            self.sample_weights['any_injury'][any_injury.tolist()]
        ])

        images, crop_kidney, crop_liver, crop_spleen = images / 255.0, crop_kidney / 255.0, crop_liver / 255.0, crop_spleen / 255.0
        images, crop_kidney, crop_liver, crop_spleen = images.half(), crop_kidney.half(), crop_liver.half(), crop_spleen.half()
        return images, crop_kidney, crop_liver, crop_spleen, label, bowel, extravasation, kidney, liver, spleen, any_injury, sample_weights
    