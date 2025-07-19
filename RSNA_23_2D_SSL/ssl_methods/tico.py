import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Custom2DCNN, Projector
from img_aug import StrongAug, WeakAug

class TiCoLoss(nn.Module):
    def __init__(self, beta = 0.9, rho = 8.0):
        super(TiCoLoss, self).__init__()
        
        self.beta = beta
        self.rho = rho

    def forward(self, C, q, k): # q: (img_num, dim), k: (img_num, dim)
        B = torch.mm(q.T, q) / q.shape[0]
        C = self.beta * C + (1 - self.beta) * B
        loss = 1 + -(q * k).sum(dim = 1).mean() + self.rho * (torch.mm(q, C) * q).sum(dim = 1).mean()
        return C, loss

class TiCo(nn.Module):
    def __init__(self, ):
        super(TiCo, self).__init__()
        self.feature_dim = 368
        self.hidden_dim = 1024 # default: 1024
        self.hidden_layer_num = 1 # numbers of layers for projector
        self.proj_dim = 1024 # default: 1024

        self.TiCoLoss = TiCoLoss()

        self.augment_fn1 = StrongAug() # strong augmentation
        self.augment_fn2 = WeakAug() # weak augmentation
        
        self.online_encoder = Custom2DCNN()
        self.offline_encoder = Custom2DCNN()

        self.init_projector()

        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data.copy_(param_on.data)
            param_off.requires_grad = False

        self.register_buffer("C", torch.zeros(4, self.proj_dim, self.proj_dim))
        
    @torch.no_grad()
    def momentum_update_key_encoder(self, momentum):
        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data = momentum * param_off.data + (1.0 - momentum) * param_on.data

    def forward(self, X): # X: (4, img_num, H, W)
        q = self.online_encoder(*[self.augment_fn1(x) for x in X])
        q = F.normalize(q, dim = -1) # q: (4, img_num, dim)

        with torch.no_grad():
            k = self.offline_encoder(*[self.augment_fn2(x) for x in X])
            k = F.normalize(k, dim = -1) # k: (4, img_num, dim)
        
        loss_all = []
        for idx, (q_idx, k_idx) in enumerate(zip(q, k)):
            C, loss = self.TiCoLoss(self.C[idx].clone().detach(), q_idx, k_idx)
            self.C[idx] = C.detach()
            loss_all.append(loss.mean())

        return torch.stack(loss_all) # loss: (4)
    
    def init_projector(self):
        proj_kargs = {"in_dim": self.feature_dim, 
                      "hidden_dim": self.hidden_dim,
                      "proj_dim": self.proj_dim, 
                      "hidden_layer_num": self.hidden_layer_num}

        self.online_encoder.full_projector = Projector(**proj_kargs)
        self.online_encoder.kidney_projector = Projector(**proj_kargs)
        self.online_encoder.liver_projector = Projector(**proj_kargs)
        self.online_encoder.spleen_projector = Projector(**proj_kargs)

        self.offline_encoder.full_projector = Projector(**proj_kargs)
        self.offline_encoder.kidney_projector = Projector(**proj_kargs)
        self.offline_encoder.liver_projector = Projector(**proj_kargs)
        self.offline_encoder.spleen_projector = Projector(**proj_kargs)
