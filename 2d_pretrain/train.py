import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (f1_score, 
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             average_precision_score,
                             multilabel_confusion_matrix)
from .ssl_methods.tico import TiCo
from ..utils import create_training_solution, score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss(label_smoothing = 0., reduction='none')

def specificity_score(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, np.where(y_pred > 0.5, 1, 0))
    spec = mcm[:, 0, 0] / (mcm[:, 0, 0] + mcm[:, 0, 1])

    return spec.mean()

def pretrain_function(model,
                      optimizer,
                      scheduler,
                      scaler,
                      loader,
                      iters_to_accumulate,
                      tico_m):
    
    model.train()
    loss_history = [[], [], [], []]
    
    pretrain_loss = 0.
    for step_idx, step in enumerate(pbar := tqdm(loader), start = 1):
        # We only need original and ROI videos for SSL pretraining
        images, crop_kidney, crop_liver, crop_spleen, *_ = step
        X = torch.stack([images, crop_kidney, crop_liver, crop_spleen]).to(device) # X: (4, batch_size, img_num, img_h, img_w)
        X = torch.transpose(X, 0, 1) # X: (batch_size, 4, img_num, img_h, img_w)

        # Forward pass
        with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16):
            if isinstance(model, TiCo): model.momentum_update_key_encoder(tico_m)
            losses = torch.stack([model(x) for x in X]) # losses: (batch_size, 4)

        # Average losses across batches
        losses = torch.mean(losses, dim = 0) # losses: (4)

        # Record loss history
        for idx, loss in enumerate(losses):
            loss_history[idx].append(loss.item())

        # Record cumulate loss
        pretrain_loss += losses.mean().item()

        # Loss
        losses = losses.mean() / iters_to_accumulate

        # Update loss
        scaler.scale(losses).backward()
        if step_idx % iters_to_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            pbar.set_postfix_str(f"lr: {lr :.2e}, train loss: {pretrain_loss / step_idx :.5f}")
    
    return pretrain_loss / len(loader), loss_history

def train_function(model,
                   optimizer,
                   scheduler,
                   scaler,
                   loader,
                   iters_to_accumulate):
    model.train()
    
    total_loss = {"bowel": 0., "extravasation": 0., "kidney": 0., "liver": 0., "spleen": 0., "any_injury": 0.}
    total_weight = {"bowel": 0., "extravasation": 0., "kidney": 0., "liver": 0., "spleen": 0., "any_injury": 0.}
    loss_history = [[] for _ in range(5)]

    for bi, sample in enumerate(pbar := tqdm(loader)):
        sample = [x.to(device) for x in sample]
        images, crop_kidney, crop_liver, crop_spleen, _, bowel, extravasation, kidney, liver, spleen, any_injury, sample_weights = sample # ground_truth: (batch_size, 6)
        
        with torch.amp.autocast("cuda", dtype = torch.bfloat16):
            logits = model(images, crop_kidney, crop_liver, crop_spleen) # logits: (6, batch_size, 2 | 3)
        ground_truth = [bowel, extravasation, kidney, liver, spleen, any_injury]

        loss = [torch.sum(criterion(logit, y)) for logit, y in zip(logits, ground_truth)]
        weighted_loss = [loss[idx] * sample_weights[:, idx] for idx in range(len(loss))]

        # Record loss
        loss_history = [loss_history[idx] + [loss[idx].item()] for idx in range(len(loss_history))]

        # loss = prediction loss
        loss = sum(loss) / 5 / iters_to_accumulate

        scaler.scale(loss).backward()
        if (bi + 1) % iters_to_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(f'lr: {lr :.6f}, loss: {loss.item() :.4f}')

        # Record weighted loss
        for idx, organ_name in enumerate(total_loss.keys()):
            total_loss[organ_name] += weighted_loss[idx].detach().cpu()
        
        for idx, organ_name in enumerate(total_weight.keys()):
            total_weight[organ_name] += sample_weights[:, idx].sum().cpu()

    for organ_name, _ in total_loss.items():
        total_loss[organ_name] /= total_weight[organ_name]

    total_loss = sum(total_loss.values()) / 6
    return loss_history

def test_function(model,
                  loader,
                  input_df,
                  temperature=1.0):

    true_df = input_df.copy()
    test_df = input_df.copy()
    model.eval()

    # Record model output
    model_output = [[] for _ in range(5)]

    # auc
    model_preds = {"bowel": [], "extravasation": [], "kidney": [], "liver": [], "spleen": [], "any_injury": []}
    model_trues = {"bowel": [], "extravasation": [], "kidney": [], "liver": [], "spleen": [], "any_injury": []}

    metrics = {"score": {"bowel": 0, "extravasation": 0, "kidney": 0, "liver": 0, "spleen": 0, "any_injury": 0},
               "auc": {"bowel": 0, "extravasation": 0, "kidney": 0, "liver": 0, "spleen": 0, "any_injury": 0},
               "map": {"bowel": 0, "extravasation": 0, "kidney": 0, "liver": 0, "spleen": 0, "any_injury": 0},
               "f1": {"bowel": 0, "extravasation": 0, "kidney": 0, "liver": 0, "spleen": 0, "any_injury": 0},
               "prec": {"bowel": 0, "extravasation": 0, "kidney": 0, "liver": 0, "spleen": 0, "any_injury": 0},
               "recall": {"bowel": 0, "extravasation": 0, "kidney": 0, "liver": 0, "spleen": 0, "any_injury": 0},
               "spec": {"bowel": 0, "extravasation": 0, "kidney": 0, "liver": 0, "spleen": 0, "any_injury": 0}}

    for bi, sample in enumerate(tqdm(loader)):
        sample = [x.to(device) for x in sample]
        images, crop_kidney, crop_liver, crop_spleen, _, bowel, extravasation, kidney, liver, spleen, any_injury, _  = sample

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype = torch.float32):
                output = model(images, crop_kidney, crop_liver, crop_spleen)

        # output: (6, batch_size, 2 | 3)
        output = [F.softmax(out.cpu(), dim = -1) / temperature for out in output[ : -1]] + [output[-1].cpu()]

        for model_out, organ_out in zip(model_output, output[ : -1]):
            model_out += organ_out.tolist()

        model_preds["bowel"].extend(output[0][:, 1].tolist())
        model_preds["extravasation"].extend(output[1][:, 1].tolist())
        model_preds["kidney"].extend(output[2].tolist())
        model_preds["liver"].extend(output[3].tolist())
        model_preds["spleen"].extend(output[4].tolist())
        model_preds["any_injury"].extend(output[5].tolist())

        model_trues["bowel"].extend(bowel.tolist())
        model_trues["extravasation"].extend(extravasation.tolist())
        model_trues["kidney"].extend(kidney.tolist())
        model_trues["liver"].extend(liver.tolist())
        model_trues["spleen"].extend(spleen.tolist())
        model_trues["any_injury"].extend(any_injury.tolist())

    # Convert predictions and true labels to numpy arrays for metrics calculation
    model_preds = {organ_name: np.array(organ_pred) for organ_name, organ_pred in model_preds.items()}
    model_trues = {organ_name: np.array(organ_true) for organ_name, organ_true in model_trues.items()}
    
    # model_output[0]: (loader_len, 2 | 3)
    model_output = np.concatenate([np.array(model_out) for model_out in model_output], axis = -1).T # model_output: (loader_len, 13)
    for idx, column_name in enumerate(test_df.columns[1 : -2]):
        test_df[column_name] = model_output[idx]

    test_score = score(create_training_solution(true_df), test_df, 'patient_id', reduction='none')

    # Record test score
    metrics["score"] = {key: round(test_score[idx], 4) for idx, key in enumerate(metrics["score"].keys())}

    for organ_name in model_preds.keys():
        multi_class = "raise" if organ_name in ["bowel", "extravasation", "any_injury"] else "ovr"
        metrics["auc"][organ_name] = roc_auc_score(model_trues[organ_name], model_preds[organ_name], multi_class = multi_class).round(4)
        metrics["map"][organ_name] = average_precision_score(model_trues[organ_name], model_preds[organ_name]).round(4)

    for organ_name in model_preds.keys():
        if organ_name in ["bowel", "extravasation", "any_injury"]: # binary classification
            model_preds[organ_name] = np.where(model_preds[organ_name] > 0.5, 1, 0)
        else: # multiclass classification
            model_preds[organ_name] = np.argmax(model_preds[organ_name], axis = 1)

    for organ_name in model_preds.keys():
        metrics["f1"][organ_name] = f1_score(model_trues[organ_name], model_preds[organ_name], average = 'macro').round(4)
        metrics["prec"][organ_name] = precision_score(model_trues[organ_name], model_preds[organ_name], zero_division = 0.0, average = 'macro').round(4)
        metrics["recall"][organ_name] = recall_score(model_trues[organ_name], model_preds[organ_name], average = 'macro').round(4)
        metrics["spec"][organ_name] = specificity_score(model_trues[organ_name], model_preds[organ_name]).round(4)

    for metric_name in metrics.keys():
        metrics[metric_name]["avg"] = np.mean(list(metrics[metric_name].values())).round(4)

    message = metrics

    return test_df, torch.tensor(test_score).mean(), message
