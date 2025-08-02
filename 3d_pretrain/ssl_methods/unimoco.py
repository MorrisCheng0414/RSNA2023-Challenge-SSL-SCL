import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model import Custom3DCNN, Custom3DCNN_Enet, Custom3DCNN_ConvNeXt, Projector
from ..img_aug import StrongAug, WeakAug

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnifiedContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(UnifiedContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1)
        # sum_pos = (y_true * torch.exp(-y_pred)).sum(1)
        sum_pos = (y_true * torch.exp(-y_pred)).sum(1) / y_true.sum(1) # changed from sum(1) to sum(1) / y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos) # original contrastive loss
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
          
class UniMoCoLoss(nn.Module):
    def __init__(self, mode = "unimoco"):
        super(UniMoCoLoss, self).__init__()
        self.pos_temp = 0.06
        self.neg_temp = 0.08
        self.criterion = UnifiedContrastive()
        self.mode = mode
        assert self.mode in ["moco", "unimoco"]
    
    def forward(self, q, k, labels, queue, label_queue):
        ## MoCo
        # pos_logits: Nx1
        pos_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) / self.pos_temp
        # neg_logits: NxK
        neg_logits = torch.einsum('nc,ck->nk', [q, queue]) / self.neg_temp
        logits = torch.cat([pos_logits, neg_logits], dim = 1)

        ## UniMoCo
        batch_size = logits.shape[0]
        # one-hot target from augmented image
        positive_target = torch.ones((batch_size, 1)).to(device)
        # find same label images from label queue, for the query with -1, all 
        # labels: (batch_size,), label_queue: (queue_size)
        if self.mode == "moco":
            targets = torch.zeros_like(neg_logits).to(device)
        elif self.mode == "unimoco":
            targets = ((labels[:, None] == label_queue[None, :]) & (labels[:, None] != -1)).float().to(device)
        
        targets = torch.cat([positive_target, targets], dim = 1)
        loss = self.criterion(logits, targets)

        return loss

class UniMoCo(nn.Module):
    def __init__(self):
        super(UniMoCo, self).__init__()
        self.mode = "unimoco"
        self.feature_dim = 368
        self.hidden_dim = 768
        self.hidden_layer_num = 1

        self.proj_dim = 256 # default: 512
        self.queue_size = 640 # default: 510 -> 768
        self.momentum = 0.999 # default: 0.995

        self.UniMoCoLoss = UniMoCoLoss()

        self.augment_fn1 = StrongAug()
        self.augment_fn2 = WeakAug()

        self.online_encoder = Custom3DCNN()
        self.offline_encoder = Custom3DCNN()

        self.init_projector()

        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data.copy_(param_on.data)
            param_off.requires_grad = False

        self.register_buffer('queue', torch.randn(4, self.proj_dim, self.queue_size))
        self.register_buffer('queue_ptr', torch.zeros((4, 1), dtype = torch.long))
        self.queue = F.normalize(self.queue, dim = 1)

        self.register_buffer('label', torch.zeros(4, self.queue_size, dtype = torch.long) - 1)
        self.register_buffer('label_ptr', torch.zeros((4, 1), dtype = torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data = self.momentum * param_off.data + (1.0 - self.momentum) * param_on.data
 
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, labels, idx):
        labels = torch.where(labels == 0, -1, labels) # To provide only one positive for "healthy" class
        
        batch_size = keys.shape[0]

        queue_ptr = int(self.queue_ptr[idx])
        label_ptr = int(self.label_ptr[idx])

        assert self.queue_size % batch_size == 0  # for simplicity

        self.queue[idx, :, queue_ptr:queue_ptr + batch_size] = keys.T
        self.label[idx, label_ptr:label_ptr + batch_size] = labels

        queue_ptr = (queue_ptr + batch_size) % self.queue_size  # move pointer
        label_ptr = (label_ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[idx][0] = queue_ptr
        self.label_ptr[idx][0] = label_ptr
       
    @torch.no_grad()  
    def batch_shuffle(self, x):
        idx_shuffle = torch.randperm(x[0].shape[0]) # x: (4, batch_size, video_dim)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return [x[i][idx_shuffle, :, :, :] for i in range(len(x))], idx_unshuffle
     
    @torch.no_grad()
    def batch_unshuffle(self, x, idx_unshuffle):
        return x[:, idx_unshuffle, :]

    def forward(self, X, labels):
        labels_T = labels.T # (4, batch_size)
        im_q = [self.augment_fn1(x) for x in X]
        im_k = [self.augment_fn2(x) for x in X]

        q = self.online_encoder(*im_q) # q: (4, batch_size, dim)
        q = F.normalize(q, dim = -1)

        with torch.no_grad():
            self.momentum_update_key_encoder()

            im_k, idx_unshuffle = self.batch_shuffle(im_k)

            k = self.offline_encoder(*im_k)
            k = F.normalize(k, dim = -1)

            k = self.batch_unshuffle(k, idx_unshuffle)

        # Compute contrastive loss for each class (full, kidney, liver, spleen)
        loss_history = []
        for idx, (q_idx, k_idx) in enumerate(zip(q, k)):
            ## MoCo loss
            organ_loss = self.UniMoCoLoss(q_idx, k_idx, 
                                          labels_T[idx], 
                                          self.queue[idx].clone().detach(), 
                                          self.label[idx].clone().detach())
            
            self.dequeue_and_enqueue(k_idx, labels_T[idx], idx)
            
            loss_history.append(organ_loss)

        return torch.stack(loss_history) # (4, batch_size)

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

# Original contrastive loss for UniMoCo
class UnifiedContrastive_ori(UnifiedContrastive):
    def __init__(self, reduction='mean'):
        super().__init__()

    def forward(self, y_pred, y_true):
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1)
        sum_pos = (y_true * torch.exp(-y_pred)).sum(1)
        # sum_pos = (y_true * torch.exp(-y_pred)).sum(1) / y_true.sum(1) # changed from sum(1) to sum(1) / y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos) # original contrastive loss
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

class UniMoCoLoss_ori(UniMoCoLoss):
    def __init__(self, mode = "unimoco"):
        super().__init__()
        self.criterion = UnifiedContrastive_ori()

class UniMoCo_Ori(UniMoCo):
    def __init__(self):
        super().__init__()
        self.UniMoCoLoss = UniMoCoLoss_ori()

# UniMoCo with EfficientNet as backbone
class UniMoCo_Enet(UniMoCo):
    def __init__(self):
        super().__init__()
        self.feature_dim = 1000
        self.hidden_dim = 2000

        self.augment_fn1 = StrongAug(s = 224)
        self.augment_fn2 = WeakAug(s = 224)

        self.online_encoder = Custom3DCNN_Enet()
        self.offline_encoder = Custom3DCNN_Enet()

        self.init_projector()

        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data.copy_(param_on.data)
            param_off.requires_grad = False

# UniMoCo with ConvNeXt as backbone
class UniMoCo_ConvNeXt(UniMoCo):
    def __init__(self):
        super().__init__()
        self.feature_dim = 1000
        self.hidden_dim = 2000

        self.augment_fn1 = StrongAug(s = 224)
        self.augment_fn2 = WeakAug(s = 224)

        self.online_encoder = Custom3DCNN_ConvNeXt()
        self.offline_encoder = Custom3DCNN_ConvNeXt()

        self.init_projector()

        for param_on, param_off in zip(self.online_encoder.parameters(), self.offline_encoder.parameters()):
            param_off.data.copy_(param_on.data)
            param_off.requires_grad = False
