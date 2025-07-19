import torchvision
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt

IMG_SIZE = 256

## Pretraining augmentations
class _place_holder(nn.Module):
    def __init__(self):
        super(_place_holder, self).__init__()
    def forward(self, x):
        return x
    
class StrongAug(nn.Module):
    def __init__(self, s = IMG_SIZE):
        super(StrongAug, self).__init__()
        self.do_random_rotate = v2.RandomRotation(
            degrees = (-30, 30), #default: (-45, 45)
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            expand = False,
            center = None,
            fill = 0
        )
        self.do_random_scale = v2.ScaleJitter(
            target_size = [s, s],
            scale_range = (0.8, 1.2), # default: (0.8, 1.2)
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            antialias = True
        )
        self.do_random_color_jitter = v2.ColorJitter(
            brightness = 0.4, # default: 0.2
            contrast = 0.4  # default: 0.2
        )
        self.blur_prob = 1.0
        self.do_random_blur = v2.GaussianBlur(
            kernel_size = (5, 5), # default (3, 3)
            sigma = (0.5, 1.5) # default: (0.1, 1.0)
        )
        self.do_random_crop = v2.RandomResizedCrop(size = [s, s], 
                                                   scale = (0.8, 1.0),
                                                   ratio = (3 / 4, 4 / 3))
        self.do_gaussian_noise = v2.GaussianNoise(mean = 0.0, 
                                                  sigma = np.random.uniform(0.01, 0.03))
        self.flip_prob = 0.5 # default: 0.5
        self.do_horizontal_flip = v2.RandomHorizontalFlip(self.flip_prob)
        self.do_vertical_flip = v2.RandomVerticalFlip(self.flip_prob)

        self.gamma_range = (0.7, 1.5)

    def forward(self, x):
        x = self.do_random_rotate(x) 

        x = self.do_random_scale(x) 
        
        x = self.do_random_crop(x)

        x = x.unsqueeze(2) # x: (batch_size, 96, 1, H, W)
        x = self.do_random_color_jitter(x)
        x = x.squeeze(2) # x: (batch_size, 96, H, W)

        x = self.do_random_blur(x)
    
        x = self.do_gaussian_noise(x)
        
        x = v2.functional.adjust_gamma(x, gamma = np.random.uniform(*self.gamma_range)) # default: (0.7, 1.5)

        x = self.do_horizontal_flip(x) 
        x = self.do_vertical_flip(x)
        return x
    
class WeakAug(StrongAug):
    def __init__(self, s = IMG_SIZE):
        super().__init__(s = IMG_SIZE)
        
        self.do_random_rotate = v2.RandomRotation(
            degrees = (-10, 10), #default: (-45, 45)
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            expand = False,
            center = None,
            fill = 0
        )
        self.do_random_scale = v2.ScaleJitter(
            target_size = [s, s],
            scale_range = (0.9, 1.1), # default: (0.8, 1.2)
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            antialias = True
        )
        self.do_random_color_jitter = v2.ColorJitter(
            brightness = 0.1, # default: 0.2
            contrast = 0.1  # default: 0.2
        )
        self.do_random_blur = v2.GaussianBlur(
            kernel_size = (3, 3), # default (3, 3)
            sigma = (0.5, 0.8) # default: (0.1, 1.0)
        )
        self.do_gaussian_noise = _place_holder()
        self.do_random_crop = _place_holder()
        self.gamma_range = (0.9, 1.1)
        self.flip_prob = 0.0

class LinclsAug(nn.Module):
    '''
    Augmentation for original end-to-end training
    '''
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

if __name__ == "__main__":
    img_path = "/tf/ShareFromHostOS/RSNA_23_256_256_png/65355/36667/liver/20.png"
    
    img = v2.PILToTensor()(Image.open(img_path).convert("L")) / 255.0
    aug_img_strong = StrongAug()(img)
    aug_img_weak = WeakAug()(img)

    fig, axs = plt.subplots(1, 3, figsize = (15, 5))

    axs[0].imshow(img.squeeze(), cmap = "gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(aug_img_strong.squeeze(), cmap = "gray")
    axs[1].set_title("Strong Augmented Image")
    axs[2].imshow(aug_img_weak.squeeze(), cmap = "gray")
    axs[2].set_title("Weak Augmented Image")

    fig.tight_layout()
    fig.savefig("/kaggle/working/RSNA_23_UniMoCo/pngs/aug_img.png", bbox_inches = "tight")
