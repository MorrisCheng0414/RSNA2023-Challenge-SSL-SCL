import torchvision
import numpy as np
import torch.nn as nn
from torchvision.transforms import v2

IMG_SIZE = 256

class _place_holder(nn.Module):
    def __init__(self):
        super(_place_holder, self).__init__()
    def forward(self, x):
        return x
    
class StrongAug(nn.Module):
    def __init__(self, s = IMG_SIZE):
        super(StrongAug, self).__init__()
        self.blur_prob = 1.0
        self.flip_prob = 0.5 # default: 0.5
        self.gamma_range = (0.9, 1.1)

        self.do_random_rotate = v2.RandomRotation(
            degrees = (-10, 10), #default: (-45, 45)
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
            expand = False,
            center = None,
            fill = 0
        )
        # self.do_random_scale = v2.ScaleJitter(
        #     target_size = [s, s],
        #     scale_range = (0.9, 1.1), # default: (0.8, 1.2)
        #     interpolation = torchvision.transforms.InterpolationMode.BILINEAR,
        #     antialias = True
        # )
        self.do_random_color_jitter = v2.RandomApply([v2.ColorJitter(
            brightness = 0.2, # default: 0.2
            contrast = 0.2  # default: 0.2
        )], p = 0.8)
        self.do_random_blur = v2.RandomApply([v2.GaussianBlur(
            kernel_size = (3, 3), # default (3, 3)
            sigma = (0.5, 0.8) # default: (0.1, 1.0)
        )], p = self.blur_prob)
        self.do_random_crop = v2.RandomResizedCrop(size = [s, s], 
                                                   scale = (0.8, 1.0),
                                                   ratio = (3 / 4, 4 / 3))
        self.do_gaussian_noise = v2.GaussianNoise(mean = 0.0, 
                                                  sigma = np.random.uniform(0.01, 0.03))
        self.do_horizontal_flip = v2.RandomHorizontalFlip(self.flip_prob)
        self.do_vertical_flip = v2.RandomVerticalFlip(self.flip_prob)


    def forward(self, x):
        x = self.do_random_rotate(x) 

        # x = self.do_random_scale(x) 
        
        x = self.do_random_crop(x)

        x = x.unsqueeze(-3) # x: (batch_size, 96, 1, H, W)
        x = self.do_random_color_jitter(x)
        x = x.squeeze(-3) # x: (batch_size, 96, H, W)

        if np.random.rand() < self.blur_prob:
            x = self.do_random_blur(x)
    
        x = self.do_gaussian_noise(x)
        
        x = v2.functional.adjust_gamma(x, gamma = np.random.uniform(*self.gamma_range)) # default: (0.7, 1.5)

        x = self.do_horizontal_flip(x) 
        x = self.do_vertical_flip(x)
        return x
    
class WeakAug(StrongAug):
    def __init__(self, s = IMG_SIZE):
        super().__init__(s = IMG_SIZE)
        self.blur_prob = 0.1
        
        self.do_gaussian_noise = _place_holder()
        # self.do_random_crop = _place_holder()

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
