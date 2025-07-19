import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Custom2DCNN, Projector
from img_aug import StrongAug, WeakAug

class VICReg(nn.Module):
    def __init__(self):
        super(VICReg, self).__init__()
        self.feature_dim = 368
        self.hidden_dim = 2048 # default: 2048
        self.hidden_layer_num = 2
        self.proj_dim = 2048 # default: 2048

        self.online_encoder = Custom2DCNN()

        self.init_projector()

        self.augment_fn1 = StrongAug()
        self.augment_fn2 = WeakAug()
        
        self.sim_coeff = 25.0 # invariance (default: 25.0)
        self.std_coeff = 25.0 # variance (default: 25.0)
        self.cov_coeff = 1.0 # covariance (default: 1.0)
        
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    def forward(self, X): # X: (4, img_num, H, W)
        batch_size = X.shape[1] # batch_size = img_num
        x_all = self.online_encoder(*[self.augment_fn1(x) for x in X]) # x_all: (4, img_num, proj_dim)
        y_all = self.online_encoder(*[self.augment_fn2(x) for x in X]) # y_all: (4, img_num, proj_dim)
        
        loss_all = []
        for x, y in zip(x_all, y_all): # x, y: (img_num, proj_dim)
            batch_size = x.shape[0]
            repr_loss = F.mse_loss(x, y)
    
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)

            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    
            cov_x = (x.T @ x) / (batch_size - 1)
            cov_y = (y.T @ y) / (batch_size - 1)
            
            cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
                self.feature_dim
            ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.feature_dim)
    
            loss = (
                self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
            )
            loss_all.append(loss)
        
        return torch.stack(loss_all) # (4)
    
    def init_projector(self):
        proj_kargs = {"in_dim": self.feature_dim,
                      "hidden_dim": self.hidden_dim,
                      "proj_dim": self.proj_dim, 
                      "hidden_layer_num": self.hidden_layer_num}
        
        self.online_encoder.full_projector = Projector(**proj_kargs)
        self.online_encoder.kidney_projector = Projector(**proj_kargs)
        self.online_encoder.liver_projector = Projector(**proj_kargs)
        self.online_encoder.spleen_projector = Projector(**proj_kargs)
