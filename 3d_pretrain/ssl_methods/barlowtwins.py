import torch
import torch.nn as nn
from ..model import Custom3DCNN, Projector
from ..img_aug import StrongAug, WeakAug

class BarlowTwins(nn.Module):
    def __init__(self):
        super(BarlowTwins, self).__init__()
        self.feature_dim = 368
        self.hidden_dim = 2048
        self.hidden_layer_num = 1
        self.proj_dim = 2048 # TODO

        self.lambd = 0.0051

        self.online_encoder = Custom3DCNN()

        self.init_projector()

        self.bn = nn.BatchNorm1d(self.proj_dim, affine = False)

        self.augment_fn1 = StrongAug()
        self.augment_fn2 = WeakAug()
        
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    def forward(self, X): # X: (4, batch_size, img_num, H, W)
        batch_size = X.shape[1]
        z1_all = self.online_encoder(*[self.augment_fn1(x) for x in X]) # x_all: (4, batch_size, proj_dim)
        z2_all = self.online_encoder(*[self.augment_fn2(x) for x in X]) # y_all: (4, batch_size, proj_dim)
        
        loss_all = []
        for z1, z2 in zip(z1_all, z2_all): # x, y: (batch_size, proj_dim)
            c = self.bn(z1).T @ self.bn(z2)
            c /= batch_size

            on_diag = ((torch.diagonal(c) - 1) ** 2).sum()
            off_diag = (self.off_diagonal(c) ** 2).sum()

            loss = on_diag + self.lambd * off_diag
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
