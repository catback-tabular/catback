import torch
import numpy as np

# ----------------------------------------------------------------------------
# 1. embed_data_mask
# ----------------------------------------------------------------------------
def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False):
    """
    Takes in categorical and continuous data along with corresponding masks, then:
      1) Offsets categorical values by model.categories_offset so they align with the embedding table.
      2) Passes categorical data through the model's embedding layer.
      3) Passes continuous data through either an MLP (if model.cont_embeddings == 'MLP').
      4) Uses mask embedding layers for both categorical and continuous features. Where a feature is masked (mask=0),
         it replaces the original embedding with a special mask embedding.
      5) (Optionally) applies position encoding for vision datasets.

    Args:
        x_categ (Tensor):    [batch_size, n_categ] raw categorical features
        x_cont (Tensor):     [batch_size, n_cont] continuous features
        cat_mask (Tensor):   [batch_size, n_categ] (1 where present, 0 where masked)
        con_mask (Tensor):   [batch_size, n_cont]  (1 where present, 0 where masked)
        model (nn.Module):   SAINT model that contains embeddings, offsets, etc.
        vision_dset (bool):  Flag indicating if we should add position encodings.

    Returns:
        x_categ (Tensor):       The offset categorical input (batch_size, n_categ).
        x_categ_enc (Tensor):   The masked/embedded categorical features (batch_size, n_categ, embed_dim).
        x_cont_enc (Tensor):    The masked/embedded continuous features (batch_size, n_cont, embed_dim).
    """
    device = x_cont.device

    # 1) Offset categorical indices so each feature's categories fall into distinct ranges
    x_categ = x_categ + model.categories_offset.type_as(x_categ)

    # 2) Embedding for categorical features
    x_categ_enc = model.embeds(x_categ)

    n1, n2 = x_cont.shape  # n1=batch_size, n2=num_cont_features
    _, n3 = x_categ.shape  # n3=num_categ_features

    # 3) Embed continuous features if the model expects them to be processed by an MLP
    if model.cont_embeddings == 'MLP':
        x_cont_enc = torch.empty(n1, n2, model.dim)  # [batch_size, n_cont, embed_dim]
        for i in range(model.num_continuous):
            # Pass each continuous column through the corresponding MLP
            x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
    else:
        # If cont_embeddings != 'MLP', we throw an exception. (In the SAINT code, this path is rarely used.)
        raise Exception('This case should not work!')

    # Move continuous embeddings to the same device (CPU or GPU)
    x_cont_enc = x_cont_enc.to(device)

    # 4) Handle masking logic:
    #    - cat_mask_temp and con_mask_temp identify which embedding index to use (masked vs unmasked).
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    # Convert them into actual mask embeddings
    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)

    # Replace original embeddings where mask=0 with the mask embedding
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    # 5) If this is a vision dataset, add position encodings to the categorical embeddings
    if vision_dset:
        # pos will be [batch_size, n3], each row is [0,1,2,...n3-1]
        pos = np.tile(np.arange(x_categ.shape[-1]), (x_categ.shape[0], 1))
        pos = torch.from_numpy(pos).to(device)

        # Retrieve positional embeddings for these positions
        pos_enc = model.pos_encodings(pos)
        x_categ_enc += pos_enc

    return x_categ, x_categ_enc, x_cont_enc


# ----------------------------------------------------------------------------
# 2. mixup_data
# ----------------------------------------------------------------------------
def mixup_data(x1, x2, lam=1.0, y=None, use_cuda=True):
    """
    Implements the 'mixup' augmentation:
      - Shuffles the batch and creates a linear interpolation between samples.

    Args:
        x1 (Tensor): First input (e.g., categorical or embedded representation).
        x2 (Tensor): Second input (e.g., continuous or embedded representation).
        lam (float): Mixing factor between 0 and 1.
        y (Tensor):  Optional labels. If provided, we also return shuffled label pairs.
        use_cuda (bool): Indicates if the device is CUDA-capable.

    Returns:
        mixed_x1 (Tensor): Mixup result for x1.
        mixed_x2 (Tensor): Mixup result for x2.
        y_a (Tensor):       Original labels.
        y_b (Tensor):       Shuffled labels (if y is provided).
    """
    batch_size = x1.size()[0]

    # Generate a random permutation of indices
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # Mix the inputs
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]

    # If labels are provided, return both original (y_a) and shuffled (y_b) for mixup loss calculation
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b
    
    # Otherwise, just return the mixed inputs
    return mixed_x1, mixed_x2


# ----------------------------------------------------------------------------
# 3. add_noise
# ----------------------------------------------------------------------------
def add_noise(x_categ, x_cont, noise_params={'noise_type': ['cutmix'], 'lambda': 0.1}):
    """
    Applies noise-like augmentations such as 'cutmix' or 'missing' to the data.

    Currently supports:
      - 'cutmix':  Randomly replaces a portion (lambda fraction) of x_categ and x_cont 
                   with entries from a different example in the batch.
      - 'missing': Randomly sets a fraction (lambda) of elements to 0.

    Args:
        x_categ (Tensor):      [batch_size, n_categ] categorical data
        x_cont (Tensor):       [batch_size, n_cont] continuous data
        noise_params (dict):   Dictionary with:
           noise_type (str or list): e.g. ['cutmix'] or 'missing'
           lambda (float): fraction indicating how much data to replace/mask

    Returns:
        If 'cutmix':
           x_categ_corr (Tensor): x_categ with some portion replaced by random other samples in the batch.
           x_cont_corr  (Tensor): x_cont with some portion replaced by random other samples in the batch.
        If 'missing':
           masked_categ (Tensor): x_categ with fraction of elements set to 0.
           masked_cont  (Tensor): x_cont with fraction of elements set to 0.
    """
    lam = noise_params['lambda']
    device = x_categ.device
    batch_size = x_categ.size()[0]

    # -----------------------------------------
    # cutmix - replace some entries with random row from the batch
    # -----------------------------------------
    if 'cutmix' in noise_params['noise_type']:
        # Shuffle the batch
        index = torch.randperm(batch_size)

        # Decide, for each entry, whether it is replaced (cat_corr=0) or not (cat_corr=1)
        cat_corr = torch.from_numpy(
            np.random.choice(2, x_categ.shape, p=[lam, 1 - lam])
        ).to(device)

        con_corr = torch.from_numpy(
            np.random.choice(2, x_cont.shape, p=[lam, 1 - lam])
        ).to(device)

        x1, x2 = x_categ[index, :], x_cont[index, :]
        x_categ_corr = x_categ.clone().detach()
        x_cont_corr  = x_cont.clone().detach()

        # Where cat_corr==0, copy from the shuffled batch example
        x_categ_corr[cat_corr == 0] = x1[cat_corr == 0]
        x_cont_corr[con_corr == 0]  = x2[con_corr == 0]
        return x_categ_corr, x_cont_corr

    # -----------------------------------------
    # missing - set fraction lam of entries to 0
    # -----------------------------------------
    elif noise_params['noise_type'] == 'missing':
        x_categ_mask = np.random.choice(2, x_categ.shape, p=[lam, 1 - lam])
        x_cont_mask  = np.random.choice(2, x_cont.shape, p=[lam, 1 - lam])

        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask  = torch.from_numpy(x_cont_mask).to(device)

        # Multiply to zero out chosen elements
        return torch.mul(x_categ, x_categ_mask), torch.mul(x_cont, x_cont_mask)

    else:
        # Future expansions or other noise types
        print("This noise type is not yet implemented.")
