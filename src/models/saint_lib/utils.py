import torch
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
from .augmentations import embed_data_mask
import torch.nn as nn

# ----------------------------------------------------------------------------
# 1. make_default_mask
# ----------------------------------------------------------------------------
def make_default_mask(x):
    """
    Creates a default mask for an input array 'x'.
    By default, it returns an array of all ones (indicating no masking) 
    except for the last column, which is set to zero.
    This might be used in certain masking tasks or 
    as a placeholder to mark the final feature.

    Args:
        x (np.ndarray): shape [N, D], where N is number of samples, D is number of features.

    Returns:
        mask (np.ndarray): shape [N, D], where mask[:,-1] = 0, else 1.
    """
    mask = np.ones_like(x)
    mask[:, -1] = 0
    return mask

# ----------------------------------------------------------------------------
# 2. tag_gen
# ----------------------------------------------------------------------------
def tag_gen(tag, y):
    """
    Given a string tag and a dictionary y with y['data'] of shape [N, 1],
    returns a string array of length N, each filled with 'tag'.
    Useful for labeling or grouping samples when logging results or 
    evaluating model performance.

    Args:
        tag (str): A label to assign to each row.
        y (dict):  A dictionary containing 'data' of shape [N, 1].

    Returns:
        np.ndarray of shape [N] with each element = tag.
    """
    return np.repeat(tag, len(y['data']))

# ----------------------------------------------------------------------------
# 3. count_parameters
# ----------------------------------------------------------------------------
def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The model whose parameters we want to count.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------------------------------------------------------
# 4. get_scheduler
# ----------------------------------------------------------------------------
def get_scheduler(args, optimizer):
    """
    Returns the appropriate learning rate scheduler based on 'args.scheduler'.
    
    Supported schedulers:
      - 'cosine': uses CosineAnnealingLR
      - 'linear': uses MultiStepLR with milestones at ~1/2.667, ~1/1.6, and ~1/1.142 of total epochs

    Args:
        args (argparse.Namespace): Contains the fields:
            scheduler (str): 'cosine' or 'linear'
            epochs (int): total epochs for training
        optimizer (torch.optim.Optimizer): The optimizer whose LR will be scheduled.

    Returns:
        torch.optim.lr_scheduler: The constructed scheduler.
    """
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'linear':
        # The three milestones effectively reduce LR at different epoch ratios.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                args.epochs // 2.667, 
                args.epochs // 1.6, 
                args.epochs // 1.142
            ],
            gamma=0.1
        )
    return scheduler

# ----------------------------------------------------------------------------
# 5. imputations_acc_justy
# ----------------------------------------------------------------------------
def imputations_acc_justy(model, dloader, device):
    """
    Evaluates the accuracy and AUROC for a specific imputation-related 
    task where the last column of x_categ is the "label" to predict. 
    The model is expected to reconstruct or predict x_categ[:, -1].

    Here we:
      1) Embed the categorical and continuous inputs with embed_data_mask().
      2) Forward pass through the model's transformer + final classification layers.
      3) Compare the predicted class vs. actual class of the last categorical feature.
      4) Return accuracy and AUROC.

    Args:
        model (nn.Module): A SAINT-like model with transformer + mlpfory.
        dloader (DataLoader): DataLoader with the needed batch structure.
        device (torch.device): 'cuda' or 'cpu'.

    Returns:
        acc (float): Accuracy in percentage (0-100).
        auc (float): AUROC score (0-1).
    """
    model.eval()
    m = nn.Softmax(dim=1)

    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            # data[0] = x_categ, data[1] = x_cont, data[2] = cat_mask, data[3] = con_mask
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)

            # Embed data
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
            # Forward pass
            reps = model.transformer(x_categ_enc, x_cont_enc)
            # For imputation tasks, they pick representation near the end of the categorical dimension
            y_reps = reps[:, model.num_categories - 1, :]  
            y_outs = model.mlpfory(y_reps)

            # The "true label" is x_categ[:, -1]
            y_test = torch.cat([y_test, x_categ[:, -1].float()], dim=0)

            # The predicted class is argmax of y_outs
            y_pred = torch.cat([y_pred, torch.argmax(m(y_outs), dim=1).float()], dim=0)

            # Probability for the last class used for AUROC
            prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)

    # Calculate accuracy
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100

    # Calculate AUROC
    auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    return acc, auc

# ----------------------------------------------------------------------------
# 6. multiclass_acc_justy
# ----------------------------------------------------------------------------
def multiclass_acc_justy(model, dloader, device):
    """
    Similar to imputations_acc_justy, but for a multiclass scenario. 
    We only compute the accuracy (and return 0 for the second metric, as 
    AUROC is not trivially defined for multiple classes unless a one-vs-all approach is used).

    Args:
        model (nn.Module): The model to evaluate.
        dloader (DataLoader): DataLoader for the multiclass task.
        device (torch.device): 'cuda' or 'cpu'.

    Returns:
        acc (float): Accuracy in percentage (0-100).
        0 (int): Placeholder for a second metric, always returns 0 here.
    """
    model.eval()
    vision_dset = True  # Possibly indicates position encodings for a vision-like task
    m = nn.Softmax(dim=1)

    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)

            # For multiclass, we apply embed_data_mask with vision_dset = True
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, model.num_categories - 1, :]
            y_outs = model.mlpfory(y_reps)

            # True label is x_categ[:, -1]
            y_test = torch.cat([y_test, x_categ[:, -1].float()], dim=0)
            # Predicted label is the argmax over classes
            y_pred = torch.cat([y_pred, torch.argmax(m(y_outs), dim=1).float()], dim=0)

    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100
    return acc, 0

# ----------------------------------------------------------------------------
# 7. classification_scores
# ----------------------------------------------------------------------------
def classification_scores(model, dloader, device, task, vision_dset):
    """
    Computes accuracy (and AUROC if it's a binary task) for standard classification.

    In SAINT, the CLS token is typically at index 0 in the sequence, 
    so we use reps[:, 0, :] for the representation of the CLS token 
    to perform classification via model.mlpfory.

    Args:
        model (nn.Module):        SAINT model for classification.
        dloader (DataLoader):     DataLoader providing x_categ, x_cont, y_gts, cat_mask, con_mask.
        device (torch.device):    'cuda' or 'cpu'.
        task (str):               'binary' or 'multiclass'.
        vision_dset (bool):       Flag for additional positional encodings if vision data.

    Returns:
        acc (float): Accuracy in percentage (0-100).
        auc (float): AUROC (only computed if task=='binary'), else 0.
    """
    model.eval()
    m = nn.Softmax(dim=1)

    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            # data[0] = x_categ, data[1] = x_cont, data[2] = y_gts, data[3] = cat_mask, data[4] = con_mask
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

            # Embed data
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)

            # Pass through transformer
            reps = model.transformer(x_categ_enc, x_cont_enc)
            # Use the CLS token representation, which is at index 0 for SAINT
            y_reps = reps[:, 0, :]
            y_outs = model.mlpfory(y_reps)

            # Store ground-truth labels
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            # Predicted class is argmax across class dimension
            y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).float()], dim=0)

            # If binary classification, store the probability for the positive class
            if task == 'binary':
                prob = torch.cat([prob, m(y_outs)[:, -1].float()], dim=0)

    # Compute accuracy
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0] * 100

    # Compute AUROC only for binary classification and if there are more than 1 class in y_true
    auc = 0
    if task == 'binary' and len(torch.unique(y_test)) > 1:
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())

    return acc.cpu().numpy(), auc

# ----------------------------------------------------------------------------
# 8. mean_sq_error
# ----------------------------------------------------------------------------
def mean_sq_error(model, dloader, device, vision_dset):
    """
    Computes the root-mean-square error (RMSE) for regression tasks.

    The typical structure is:
      1) Embed the categorical and continuous features.
      2) Pass them through the transformer to get representations.
      3) Use the CLS token's representation (index 0) for regression via mlpfory.
      4) Compare predictions y_outs with the ground truth y_gts 
         using mean_squared_error (sklearn) with squared=False -> RMSE.

    Args:
        model (nn.Module):      SAINT model set up for regression.
        dloader (DataLoader):   DataLoader containing x_categ, x_cont, y_gts, cat_mask, con_mask.
        device (torch.device):  'cuda' or 'cpu'.
        vision_dset (bool):     Flag for additional positional encodings if used.

    Returns:
        rmse (float): Root Mean Squared Error across all samples in the data loader.
    """
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            # data[0], data[1] are cat/cont inputs; data[2] is ground truth y; data[3], data[4] are masks
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

            # Embed
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)

            # Forward pass
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]            # use CLS token
            y_outs = model.mlpfory(y_reps)    # regression output

            # Accumulate ground truth and predictions
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs], dim=0)

        # Compute RMSE via sklearn's mean_squared_error with squared=False
        rmse = mean_squared_error(y_test.cpu(), y_pred.cpu(), squared=False)
        return rmse
