import timm
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, convnext_tiny
from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormModel

class FeatureExtractor(nn.Module):
    def __init__(self, hidden, num_channel):
        super(FeatureExtractor, self).__init__()

        self.hidden = hidden
        self.num_channel = num_channel

        self.cnn = timm.create_model(model_name = 'regnety_002',
                                     pretrained = True,
                                     num_classes = 0,
                                     in_chans = num_channel)

        self.fc = nn.Linear(hidden, hidden//2)

    def forward(self, x):
        if len(x.shape) == 3:
            batch_size = 1
            num_frame, h, w = x.shape
        else:
            batch_size, num_frame, h, w = x.shape
        if num_frame % self.num_channel != 0:
            x = x[:, 0 : -(num_frame % self.num_channel), :, :]
            
        x = x.reshape(batch_size, num_frame//self.num_channel, self.num_channel, h, w)
        x = x.reshape(-1, self.num_channel, h, w)
        x = self.cnn(x)
        x = x.reshape(batch_size, num_frame//self.num_channel, self.hidden)

        x = self.fc(x)
        return x

class ContextProcessor(nn.Module):
    def __init__(self, hidden):
        super(ContextProcessor, self).__init__()
        self.transformer = RobertaPreLayerNormModel(
            RobertaPreLayerNormConfig(
                hidden_size = hidden//2,
                num_hidden_layers = 1,
                num_attention_heads = 4,
                intermediate_size = hidden*2,
                hidden_act = 'gelu_new',
                )
            )

        del self.transformer.embeddings.word_embeddings

        self.dense = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.transformer(inputs_embeds = x).last_hidden_state

        apool = torch.mean(x, dim = 1)
        mpool, _ = torch.max(x, dim = 1)
        x = torch.cat([mpool, apool], dim = -1)

        x = self.dense(x)
        x = self.activation(x)
        return x

class Custom3DCNN(nn.Module):
    def __init__(self, hidden = 368, num_channel = 3):
        super(Custom3DCNN, self).__init__()

        self.full_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.kidney_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.liver_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)
        self.spleen_extractor = FeatureExtractor(hidden=hidden, num_channel=num_channel)

        self.full_processor = ContextProcessor(hidden=hidden)
        self.kidney_processor = ContextProcessor(hidden=hidden)
        self.liver_processor = ContextProcessor(hidden=hidden)
        self.spleen_processor = ContextProcessor(hidden=hidden)

        self.full_projector = torch.nn.Identity()
        self.kidney_projector = torch.nn.Identity()
        self.liver_projector = torch.nn.Identity()
        self.spleen_projector = torch.nn.Identity()

    def forward(self, full_input, crop_kidney, crop_liver, crop_spleen):
        full_output = self.full_extractor(full_input)
        kidney_output = self.kidney_extractor(crop_kidney)
        liver_output = self.liver_extractor(crop_liver)
        spleen_output = self.spleen_extractor(crop_spleen)

        full_output2 = self.full_processor(torch.cat([full_output, kidney_output, liver_output, spleen_output], dim = 1))
        kidney_output2 = self.kidney_processor(torch.cat([full_output, kidney_output], dim = 1))
        liver_output2 = self.liver_processor(torch.cat([full_output, liver_output], dim = 1))
        spleen_output2 = self.spleen_processor(torch.cat([full_output, spleen_output], dim = 1))

        full_projection = self.full_projector(full_output2)
        kidney_projection = self.kidney_projector(kidney_output2)
        liver_projection = self.liver_projector(liver_output2)
        spleen_projection = self.spleen_projector(spleen_output2)

        return torch.stack([full_projection, kidney_projection, liver_projection, spleen_projection])
        
class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, hidden_layer_num = 1):
        super(Projector, self).__init__()
        layers = []
        f = [in_dim] + [hidden_dim] * hidden_layer_num # [in_dim, hidden_dim * n]
        for i in range(len(f) - 1):
            layers.append(nn.Linear(f[i], f[i+1]))
            layers.append(nn.BatchNorm1d(f[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(f[-1], proj_dim, bias = False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x
    
class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, hidden_layer_num = 1):
        super(Predictor, self).__init__()
        pred_kargs = {"in_dim": in_dim, 
                      "hidden_dim": hidden_dim, 
                      "proj_dim": proj_dim, 
                      "hidden_layer_num": hidden_layer_num}
        
        self.full_predictor = Projector(**pred_kargs)
        self.kidney_predictor = Projector(**pred_kargs)
        self.liver_predictor = Projector(**pred_kargs)
        self.spleen_predictor = Projector(**pred_kargs)

    def forward(self, full_proj, kidney_proj, liver_proj, spleen_proj):
        full_proj = self.full_predictor(full_proj)
        kidney_proj = self.kidney_predictor(kidney_proj)
        liver_proj = self.liver_predictor(liver_proj)
        spleen_proj = self.spleen_predictor(spleen_proj)

        return torch.stack([full_proj, kidney_proj, liver_proj, spleen_proj])

class Classifier(nn.Module):
    def __init__(self, in_dim = 368):
        super(Classifier, self).__init__()
        self.bowel = nn.Linear(in_dim, 2)
        self.extra = nn.Linear(in_dim, 2)
        self.kidney = nn.Linear(in_dim, 3)
        self.liver = nn.Linear(in_dim, 3)
        self.spleen = nn.Linear(in_dim, 3)

    def forward(self, full_output, kidney_output, liver_output, spleen_output):
        bowel = self.bowel(full_output)
        extra = self.extra(full_output)
        kidney = self.kidney(kidney_output)
        liver = self.liver(liver_output)
        spleen = self.spleen(spleen_output)

        return bowel, extra, kidney, liver, spleen
    
class MergeModel(nn.Module):
    def __init__(self, in_dim = 368):
        super(MergeModel, self).__init__()
        self.extractor = Custom3DCNN(hidden = in_dim)
        self.classifier = Classifier(in_dim = in_dim)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, full_input, crop_kidney, crop_liver, crop_spleen):
        embeds = self.extractor(full_input, crop_kidney, crop_liver, crop_spleen)
        bowel, extra, kidney, liver, spleen = self.classifier(*embeds)

        any_injury = torch.stack([
            self.softmax(bowel)[:, 0],
            self.softmax(extra)[:, 0],
            self.softmax(kidney)[:, 0],
            self.softmax(liver)[:, 0],
            self.softmax(spleen)[:, 0]
        ], dim = -1)
        any_injury = 1 - any_injury
        any_injury, _ = any_injury.max(1)
        return bowel, extra, kidney, liver, spleen, any_injury

## EfficientNet based models
class FeatureExtractor_Enet(FeatureExtractor):
    def __init__(self, hidden, num_channel):
        super().__init__(hidden, num_channel)
        self.cnn = efficientnet_b0(weights = "DEFAULT")
        
class Custom3DCNN_Enet(Custom3DCNN):
    def __init__(self, hidden = 1000, num_channel = 3):
        super().__init__(hidden = hidden, num_channel = num_channel)
        self.full_extractor = FeatureExtractor_Enet(hidden=hidden, num_channel=num_channel)
        self.kidney_extractor = FeatureExtractor_Enet(hidden=hidden, num_channel=num_channel)
        self.liver_extractor = FeatureExtractor_Enet(hidden=hidden, num_channel=num_channel)
        self.spleen_extractor = FeatureExtractor_Enet(hidden=hidden, num_channel=num_channel)

class MergeModel_Enet(MergeModel):
    def __init__(self, in_dim = 1000):
        super().__init__(in_dim = in_dim)
        self.extractor = Custom3DCNN_Enet(hidden = in_dim)

## ConvNeXt based models
class FeatureExtractor_ConvNeXt(FeatureExtractor):
    def __init__(self, hidden, num_channel):
        super().__init__(hidden, num_channel)
        self.cnn = convnext_tiny(weights = "DEFAULT")
        
class Custom3DCNN_ConvNeXt(Custom3DCNN):
    def __init__(self, hidden = 1000, num_channel = 3):
        super().__init__(hidden = hidden, num_channel = num_channel)
        self.full_extractor = FeatureExtractor_ConvNeXt(hidden=hidden, num_channel=num_channel)
        self.kidney_extractor = FeatureExtractor_ConvNeXt(hidden=hidden, num_channel=num_channel)
        self.liver_extractor = FeatureExtractor_ConvNeXt(hidden=hidden, num_channel=num_channel)
        self.spleen_extractor = FeatureExtractor_ConvNeXt(hidden=hidden, num_channel=num_channel)

class MergeModel_ConvNeXt(MergeModel):
    def __init__(self, in_dim = 1000):
        super().__init__(in_dim = in_dim)
        self.extractor = Custom3DCNN_ConvNeXt(hidden = in_dim)

