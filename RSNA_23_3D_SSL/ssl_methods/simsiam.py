import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Custom3DCNN, Projector, Predictor
from img_aug import StrongAug, WeakAug

class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()
        self.feature_dim = 368
        self.hidden_dim = 2048
        self.hidden_layer_num = 1 # numbers of layers for projector and predictor
        self.proj_dim = 2048

        self.augment_fn1 = StrongAug() # strong augmentation
        self.augment_fn2 = WeakAug() # weak augmentation
        
        self.online_encoder = Custom3DCNN() # encoder_q

        self.init_projector()

        self.predictor = Predictor(in_dim = self.hidden_dim, 
                                   hidden_dim = self.hidden_dim,
                                   proj_dim = self.proj_dim, 
                                   hidden_layer_num = self.hidden_layer_num)

    def loss_fn(self, p, z):
        '''
        Cosine Similarity Loss
        '''
        return torch.stack([(F.cosine_similarity(p_idx, z_idx, dim = 1)).mean() for p_idx, z_idx in zip(p, z)])

    def forward(self, X): # X: (4, batch_size, img_num, H, W)
        X_view1 = [self.augment_fn1(x) for x in X]
        X_view2 = [self.augment_fn2(x) for x in X]

        proj_1 = self.online_encoder(*X_view1) # proj: (4, batch_size, proj_dim)
        proj_2 = self.online_encoder(*X_view2)

        pred_1 = self.predictor(*proj_1) # pred: (4, batch_size, proj_dim)
        pred_2 = self.predictor(*proj_2)

        loss = self.loss_fn(pred_1, proj_2.detach()) + self.loss_fn(pred_2, proj_1.detach()) # loss: (4, batch_size)
        loss = -loss / 2
        
        return loss # loss: (4)
    
    def init_projector(self):
        proj_kargs = {"in_dim": self.feature_dim, 
                      "hidden_dim": self.hidden_dim,
                      "proj_dim": self.proj_dim, 
                      "hidden_layer_num": self.hidden_layer_num}

        self.online_encoder.full_projector = Projector(**proj_kargs)
        self.online_encoder.kidney_projector = Projector(**proj_kargs)
        self.online_encoder.liver_projector = Projector(**proj_kargs)
        self.online_encoder.spleen_projector = Projector(**proj_kargs)