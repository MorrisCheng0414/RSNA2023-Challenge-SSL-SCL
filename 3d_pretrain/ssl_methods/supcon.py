import torch
import torch.nn as nn
import torch.nn.functional as F
from .unimoco import UniMoCo, UniMoCo_Enet, UniMoCo_ConvNeXt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SupConLoss(nn.Module):
    def __init__(self):
        super(SupConLoss, self).__init__()
        self.temp = 0.07
    
    def forward(self, q, k, labels, queue, label_queue):
        batch_size = labels.shape[0]
        # one-hot target from augmented image
        positive_target = torch.ones((batch_size, 1)).to(device)
        # find same label images from label queue, for the query with -1, all 
        # labels: (batch_size,), label_queue: (queue_size)
        targets = ((labels[:, None] == label_queue[None, :]) & (labels[:, None] != -1)).float().to(device)
        targets = torch.cat([positive_target, targets], dim = 1)

        ## MoCo
        # pos_logits: Nx1
        pos_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # neg_logits: NxK
        neg_logits = torch.einsum('nc,ck->nk', [q, queue])
        logits = torch.cat([pos_logits, neg_logits], dim = 1) / self.temp

        ## SupCon
        # For numerical stability
        logits_max, _ = torch.max(logits, dim = 1, keepdim = True)
        scaled_logits = logits - logits_max.detach()

        # Compute the log-probabilities
        # exp_logits = torch.exp(scaled_logits)
        # log_prob = scaled_logits - torch.log(exp_logits.sum(dim = 1, keepdim = True))
        log_prob = F.log_softmax(scaled_logits, dim = -1)

        # Nx(1+K) -> N
        mean_log_prob_pos = (log_prob * targets).sum(dim = 1) / targets.sum(dim = 1).clamp(min = 1.0) # clamp to avoid 0-division
        
        loss = -mean_log_prob_pos
        # N -> 1
        loss = loss.mean()
        
        return loss

class SupCon(UniMoCo):
    def __init__(self):
        super().__init__()
        self.UniMoCoLoss = SupConLoss()

class SupCon_Enet(UniMoCo_Enet):
    def __init__(self):
        super().__init__()
        self.UniMoCoLoss = SupConLoss()

class SupCon_ConvNeXt(UniMoCo_ConvNeXt):
    def __init__(self):
        super().__init__()
        self.UniMoCoLoss = SupConLoss()