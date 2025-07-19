import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Custom2DCNN, Projector, Predictor
from img_aug import StrongAug, WeakAug

class BYOL(nn.Module):
    def __init__(self):
        super(BYOL, self).__init__()
        self.feature_dim = 368
        self.hidden_dim = 2048 # default: 2048
        self.hidden_layer_num = 1 

        self.proj_dim = 2048 # default: 2048
        self.momentum = 0.99

        self.augment_fn1 = StrongAug() # strong augmentation
        self.augment_fn2 = WeakAug() # weak augmentation
        
        self.online_encoder = Custom2DCNN() # encoder_q
        self.offline_encoder = Custom2DCNN() # encoder_k

        self.init_projector()

        self.predictor = Predictor(in_dim = self.hidden_dim, 
                                   hidden_dim = self.hidden_dim,
                                   proj_dim = self.proj_dim, 
                                   hidden_layer_num = self.hidden_layer_num)

        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data.copy_(param_on.data)
            param_off.requires_grad = False

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data = self.momentum * param_off.data + (1.0 - self.momentum) * param_on.data
    
    def loss_fn(self, x, y):
        '''
        L2 Loss
        '''
        x = F.normalize(x, dim = -1, p = 2)
        y = F.normalize(y, dim = -1, p = 2)
        return torch.stack([2 - 2 * (x_idx * y_idx).sum(dim = -1) for x_idx, y_idx in zip(x, y)])

    def forward(self, X): # X: (4, img_num, H, W)
        X_view1 = [self.augment_fn1(x) for x in X]
        X_view2 = [self.augment_fn2(x) for x in X]

        pred_1 = self.predictor(*self.online_encoder(*X_view1)) # pred: (4, img_num, proj_dim)
        pred_2 = self.predictor(*self.online_encoder(*X_view2))

        with torch.no_grad():
            self.momentum_update_key_encoder()
            proj_1 = self.offline_encoder(*X_view1) # proj: (4, img_num, proj_dim)
            proj_2 = self.offline_encoder(*X_view2)

        loss = self.loss_fn(pred_1, proj_2.detach()) + self.loss_fn(pred_2, proj_1.detach()) # loss: (4, img_num)

        return loss.mean(dim = -1) # loss: (4)
    
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