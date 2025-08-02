import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model import Custom3DCNN, Projector
from ..img_aug import StrongAug, WeakAug

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MoCoLoss(nn.Module):
    def __init__(self):
        super(MoCoLoss, self).__init__()
        self.pos_temp = 0.06
        self.neg_temp = 0.07
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, q, k, queue): # q: (img_num, dim), k: (img_num, dim), queue: (dim, queue_size)
        # pos_logits: Nx1
        pos_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) / self.pos_temp
        # neg_logits: NxK
        neg_logits = torch.einsum('nc,ck->nk', [q, queue]) / self.neg_temp

        logits = torch.cat([pos_logits, neg_logits], dim = 1)

        labels = torch.zeros(logits.shape[0], dtype = torch.long).to(device) # (batch_size,)

        loss = self.criterion(logits, labels)

        return loss

class MoCo(nn.Module):
    def __init__(self):
        super(MoCo, self).__init__()
        self.feature_dim = 368
        self.hidden_dim = 768
        self.hidden_layer_num = 1 # numbers of layers for projector

        self.proj_dim = 256 # default: 512
        self.queue_size = 640 # default: 640
        self.momentum = 0.995 # default: 0.995

        self.MoCoLoss = MoCoLoss()

        self.augment_fn1 = StrongAug() # strong augmentation
        self.augment_fn2 = WeakAug() # weak augmentation

        self.online_encoder = Custom3DCNN() # encoder_q
        self.offline_encoder = Custom3DCNN() # encoder_k

        self.init_projector()

        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data.copy_(param_on.data)
            param_off.requires_grad = False

        self.register_buffer('queue', torch.randn(4, self.proj_dim, self.queue_size))
        self.queue = F.normalize(self.queue, dim = 1) # normalize along projection dimension
        
        self.register_buffer('queue_ptr', torch.zeros((4, 1), dtype = torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data = self.momentum * param_off.data + (1.0 - self.momentum) * param_on.data
 
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, idx):
        batch_size = keys.shape[0]

        queue_ptr = int(self.queue_ptr[idx])
        assert self.queue_size % batch_size == 0  # for simplicity

        self.queue[idx, :, queue_ptr:queue_ptr + batch_size] = keys.T
        queue_ptr = (queue_ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[idx][0] = queue_ptr
       
    @torch.no_grad()  
    def batch_shuffle(self, x):
        idx_shuffle = torch.randperm(x[0].shape[0]) # x: (4, batch_size, video_dim
        idx_unshuffle = torch.argsort(idx_shuffle)
        return [x[i][idx_shuffle, :, :, :] for i in range(len(x))], idx_unshuffle
     
    @torch.no_grad()
    def batch_unshuffle(self, x, idx_unshuffle): # x: (4, batch_size, dim)
        return x[:, idx_unshuffle, :]

    def forward(self, X): # X: (4, batch_size, img_num, H, W)
        im_q = [self.augment_fn1(x) for x in X]
        im_k = [self.augment_fn2(x) for x in X]

        q = self.online_encoder(*im_q) # q: (4, batch_size, dim)
        q = F.normalize(q, dim = -1)

        with torch.no_grad():
            self.momentum_update_key_encoder()

            im_k, idx_unshuffle = self.batch_shuffle(im_k)

            k = self.offline_encoder(*im_k)
            k = F.normalize(k, dim = -1) # k: (4, batch_size, dim)

            k = self.batch_unshuffle(k, idx_unshuffle)

        # Compute contrastive loss for each class (full, kidney, liver, spleen)
        loss_history = [] # loss: (4, img_num)
        for idx, (q_idx, k_idx) in enumerate(zip(q, k)): # q_idx, k_idx: (img_num, dim)
            organ_loss = self.MoCoLoss(q_idx, 
                                       k_idx, 
                                       self.queue[idx].clone().detach())
            
            self.dequeue_and_enqueue(k_idx, idx)
            
            loss_history.append(organ_loss)

        return torch.stack(loss_history) # loss: (4)
    
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
