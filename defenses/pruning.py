# pruning.py
"""
Fine-Pruning Defense
===============================================================

This file implements the Fine-Pruning defense against backdoor attacks.
It prunes neurons with the lowest average activation on clean data
(which are hypothesized to encode backdoor behavior) and optionally
fine-tunes the pruned model on a small subset of clean data.

Supports pruning for:
 - FT-Transformer (feed-forward layer pruning)
 - SAINT (feed-forward layer + CLS token output pruning)

Note:
 - You must adapt model and dataset paths to your actual setup.
"""

import gc
from pathlib import Path
import sys
import os
import csv

# Add the project root to the Python path
# Get the absolute path of the current file
current_file = Path(__file__).resolve()
# Get the project root (two directories up from the current file)
project_root = current_file.parent.parent.parent
# Add the project root to sys.path
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.attack import Attack




# importing all the models for this project
from src.models.FTT import FTTModel
from src.models.Tabnet import TabNetModel
from src.models.SAINT import SAINTModel
from src.models.CatBoost import CatBoostModel
from src.models.XGBoost import XGBoostModel

# importing all the datasets for this project
from src.dataset.BM import BankMarketing
from src.dataset.ACI import ACI
from src.dataset.HIGGS import HIGGS
from src.dataset.Diabetes import Diabetes
from src.dataset.Eye_Movement import EyeMovement
from src.dataset.KDD99 import KDD99
from src.dataset.CreditCard import CreditCard
from src.dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py
from src.dataset.Poker import Poker



################ <BOS> the functions and classes for pruning FT_Transformer models ################
###########################################################################################

def register_ff_hooks(model):
    """
    Register forward hooks on the last linear layer of each feed-forward network in the transformer.
    
    Args:
        model (FTTransformer): The transformer model
        
    Returns:
        dict: Dictionary mapping layer names to their activations
        list: List of hook handles for later removal if needed
    """
    activations = {}
    hooks = []
    
    # Function to capture activations
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks on the last linear layer of each feed-forward network
    for layer_idx, layer in enumerate(model.transformer.layers):
        # The feed-forward network is the second component (index 1) of each transformer layer
        # The last linear layer is at index 4 of the sequential feed-forward module
        ff_network = layer[1]  # Get the feed-forward network
        last_linear = ff_network[4]  # Get the last linear layer (projection back to original dim)
        
        # Register the hook
        layer_name = f"ff_layer_{layer_idx}_output"
        # add the layer name to the activations dictionary keys
        activations[layer_name] = None
        hook = last_linear.register_forward_hook(get_activation(layer_name))
        hooks.append(hook)
        
        
    return activations, hooks


def get_ff_layer_activations_ftt(
    ftt_model,          # An instance of FTTModel
    dataset,            # A PyTorch Dataset yielding (X_c, X_n, y)
    batch_size=32,
    device=None,
    model_type="original"  
):
    """
    Collects activations from the last linear layer of each feed-forward network
    in the transformer model for all samples in the dataset.

    Args:
        ftt_model (FTTModel): your FTT model wrapper
        dataset (Dataset): yields (X_c, X_n, y)
        batch_size (int): batch size for collection
        device (torch.device): CPU/GPU device
        model_type (str): "original" or "converted"; whichever you are analyzing

    Returns:
        dict: Dictionary mapping layer names to their activations across all samples
              Each entry has shape [num_samples, dim]
    """


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    ftt_model.to(device, model_type=model_type)
    ftt_model.eval(model_type=model_type)

    # Get the inner model
    inner_model = ftt_model.model_original if model_type == 'original' else ftt_model.model_converted

    for name, param in inner_model.named_parameters():
        print(name, param.shape)
    
    # Register hooks on feed-forward layers
    ff_activations, hooks = register_ff_hooks(inner_model)
    print(ff_activations)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Dictionary to store activations for each layer across all batches
    all_activations = {layer_name: [] for layer_name in ff_activations.keys()}

    print(all_activations)


    # Collect activations with no gradients
    with torch.no_grad():
        if model_type == 'original':
            for X_c, X_n, _ in loader:
                X_c = X_c.to(device)
                X_n = X_n.to(device)

                # Forward pass to trigger hooks
                _ = ftt_model.forward_original(X_c, X_n)
                
                # Store activations from this batch
                for layer_name, activation in ff_activations.items():
                    all_activations[layer_name].append(activation.cpu())
        
        elif model_type == 'converted':
            for X_n, _ in loader:
                # X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                # X_c, X_n = X_c.to(device), X_n.to(device)
                X_n = X_n.to(device)
                
                # Forward pass to trigger hooks
                _ = ftt_model.forward(X_n)
                
                # Store activations from this batch
                for layer_name, activation in ff_activations.items():
                    all_activations[layer_name].append(activation.cpu())
    
    # Remove hooks to free memory
    for hook in hooks:
        hook.remove()

    
    del loader
    del dataset
    gc.collect()
    # Concatenate activations from all batches for each layer
    for layer_name in all_activations.keys():
        all_activations[layer_name] = torch.cat(all_activations[layer_name], dim=0)
        print(f"{layer_name} activations shape: {all_activations[layer_name].shape}")
        gc.collect()
    
    
    
    return all_activations


def analyze_ff_activations(ff_activations, prune_ratio):
    """
    Analyzes feed-forward layer activations and identifies neurons for pruning
    based on their average activation magnitude.
    
    Args:
        ff_activations (dict): Dictionary of feed-forward layer activations
                              from get_ff_layer_activations_ftt
        prune_ratio (float): Proportion of neurons to prune (0.0 to 1.0)
        
    Returns:
        dict: Dictionary mapping layer names to indices of neurons to prune
    """
    prune_indices = {}
    
    for layer_name, activations in ff_activations.items():
        # Calculate mean activation across all samples for each neuron
        mean_activations = torch.mean(torch.abs(activations), dim=0)  # [dim]
        print(mean_activations.shape)

        # calculate the mean activation across all tokens
        mean_activations_across_tokens = torch.mean(mean_activations, dim=0)
        print(mean_activations_across_tokens.shape)
        
        # Sort neurons by activation magnitude
        sorted_indices = torch.argsort(mean_activations_across_tokens)
        print(sorted_indices.shape)
        
        # Select the bottom prune_ratio% of neurons (those with lowest activation)
        num_to_prune = int(prune_ratio * len(sorted_indices))
        indices_to_prune = sorted_indices[:num_to_prune].tolist()

        # print d type and shape of indices_to_prune
        print("type of indices_to_prune: ", type(indices_to_prune))
        print("length of indices_to_prune: ", len(indices_to_prune))
        print("indices_to_prune: ", indices_to_prune)
        
        
        
        prune_indices[layer_name] = indices_to_prune
        
        print(f"{layer_name}: Pruning {num_to_prune}/{len(sorted_indices)} neurons")
    print("prune_indices dictionary: ", prune_indices)
    
    
    return prune_indices

def create_ff_pruning_masks(ff_prune_indices, model, device=None):
    """
    Creates binary masks for feed-forward layers based on pruning indices.
    
    Args:
        ff_prune_indices (dict): Dictionary mapping layer names to indices to prune
        model (FTTransformer): The transformer model
        device (torch.device): Device to place masks on
        
    Returns:
        dict: Dictionary mapping layer names to binary masks
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pruning_masks = {}
    
    for layer_idx, layer in enumerate(model.transformer.layers):
        layer_name = f"ff_layer_{layer_idx}_output"
        
        if layer_name in ff_prune_indices:
            # Get the last linear layer of the feed-forward network
            ff_network = layer[1]
            last_linear = ff_network[4]
            
            # Create a mask of ones with the same shape as the layer output
            output_dim = last_linear.weight.shape[0]
            print("output_dim for layer: ", layer_name, " is: ", output_dim, last_linear.weight.shape)
            mask = torch.ones(output_dim, device=device)
            
            # Set pruned indices to zero
            mask[ff_prune_indices[layer_name]] = 0.0
            
            pruning_masks[layer_name] = mask
    
    return pruning_masks


class PrunedFeedForwardFTT(nn.Module):
    """
    A wrapper module that applies pruning masks to feed-forward layer outputs
    in an FTTransformer model.
    """
    def __init__(self, original_model, ff_pruning_masks):
        """
        Args:
            original_model (FTTransformer): The original transformer model
            ff_pruning_masks (dict): Dictionary mapping layer names to pruning masks
        """
        super().__init__()
        self.model = original_model
        self.ff_pruning_masks = ff_pruning_masks
        
        # Register hooks to apply masks during forward pass
        self.hooks = []
        self._register_pruning_hooks()
    
    def _register_pruning_hooks(self):
        """Register forward hooks to apply pruning masks"""
        for layer_idx, layer in enumerate(self.model.transformer.layers):
            layer_name = f"ff_layer_{layer_idx}_output"
            
            if layer_name in self.ff_pruning_masks:
                # Get the feed-forward network and its last linear layer
                ff_network = layer[1]
                last_linear = ff_network[4]
                
                # Create a hook to apply the mask
                def get_pruning_hook(mask):
                    def hook(module, input, output):
                        return output * mask
                    return hook
                
                # Register the hook
                mask = self.ff_pruning_masks[layer_name]
                hook = last_linear.register_forward_hook(get_pruning_hook(mask))
                self.hooks.append(hook)
    
    def forward(self, x_categ, x_numer):
        """
        Forward pass with pruning masks applied to feed-forward outputs.
        
        Args:
            x_categ: Categorical inputs
            x_numer: Numerical inputs
            return_attn: Whether to return attention weights
            
        Returns:
            Model output with pruning applied
        """
        return self.model(x_categ, x_numer)
    
    def remove_hooks(self):
        """Remove all hooks to free memory"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Example usage function
def prune_ff_layers_example(ftt_model, dataset, prune_ratio, model_type):
    """
    Example function showing how to use the feed-forward pruning functionality.
    
    Args:
        ftt_model: The FTT model wrapper
        dataset: Dataset for collecting activations
        prune_ratio: Proportion of neurons to prune
        model_type: "original" or "converted"
        
    Returns:
        Pruned model
    """
    # Get the inner model
    inner_model = ftt_model.model_original if model_type == 'original' else ftt_model.model_converted
    
    # 1. Collect feed-forward activations
    ff_activations = get_ff_layer_activations_ftt(ftt_model, dataset, model_type=model_type)
    
    # 2. Analyze activations and get pruning indices
    ff_prune_indices = analyze_ff_activations(ff_activations, prune_ratio)

    # 3. Create pruning masks
    ff_pruning_masks = create_ff_pruning_masks(ff_prune_indices, inner_model)
    
    # 4. Create pruned model
    pruned_model = PrunedFeedForwardFTT(inner_model, ff_pruning_masks)

    if model_type == "original":
        ftt_model.model_original = pruned_model
    else:
        ftt_model.model_converted = pruned_model
    
    return ftt_model



###########################################################################################
################ <EOS> the functions and classes for pruning FT_Transformer models ################
###########################################################################################



################ <BOS> the functions and classes for pruning SAINT models ################
########################################################################################### 

def register_saint_ff_hooks(model):
    """
    Register forward hooks on the last linear layer of each feed-forward network in SAINT.
    
    Args:
        model: The SAINT model
        
    Returns:
        dict: Dictionary mapping layer names to their activations
        list: List of hook handles for later removal if needed
    """
    activations = {}
    hooks = []
    
    # Function to capture activations
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks on the feed-forward networks in the transformer
    # First feed-forward network (column-wise)
    ff_network_1 = model.transformer.layers[0][1].fn.fn.net
    last_linear_1 = ff_network_1[3]  # Last linear layer in the first FF network
    layer_name_1 = "ff_layer_col_output"
    activations[layer_name_1] = None
    hook_1 = last_linear_1.register_forward_hook(get_activation(layer_name_1))
    hooks.append(hook_1)
    
    # Second feed-forward network (row-wise)
    ff_network_2 = model.transformer.layers[0][3].fn.fn.net
    last_linear_2 = ff_network_2[3]  # Last linear layer in the second FF network
    layer_name_2 = "ff_layer_row_output"
    activations[layer_name_2] = None
    hook_2 = last_linear_2.register_forward_hook(get_activation(layer_name_2))
    hooks.append(hook_2)
    
    return activations, hooks

def get_saint_ff_activations(saint_model, dataset, batch_size=32, device=None):
    """
    Collects activations from the feed-forward networks in SAINT.
    
    Args:
        saint_model: The SAINT model
        dataset: Dataset for collecting activations
        batch_size: Batch size for processing
        device: Computation device
        
    Returns:
        dict: Dictionary of feed-forward layer activations
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    saint_model.to(device)
    saint_model.eval()
    
    # Register hooks
    ff_activations, hooks = register_saint_ff_hooks(saint_model.model)

    print("ff_activations: ", ff_activations)
    print("hooks: ", hooks)
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Dictionary to store activations
    all_activations = {layer_name: [] for layer_name in ff_activations.keys()}
    
    # Collect activations
    with torch.no_grad():
        for batch in loader:
            # Process batch according to SAINT's input format
            inputs, targets = batch
            inputs = inputs.to(device)
            
            # Forward pass
            _ = saint_model.forward(inputs)
            
            # Store activations
            for layer_name, activation in ff_activations.items():
                all_activations[layer_name].append(activation.cpu())
                # print(f"{layer_name} activation shape: {activation.shape}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate activations
    for layer_name in all_activations.keys():
        if layer_name == "ff_layer_row_output":
            # For row output, concatenate along batch dimension (dim=1)
            stacked = torch.cat(all_activations[layer_name], dim=1)
            all_activations[layer_name] = stacked.permute(1, 0, 2)
        else:
            # For column output, concatenate along first dimension (dim=0)
            all_activations[layer_name] = torch.cat(all_activations[layer_name], dim=0)
        
        print(f"{layer_name} activations shape: {all_activations[layer_name].shape}")
    
    return all_activations

def create_saint_pruning_masks(ff_prune_indices, model, device=None):
    """
    Creates binary masks for SAINT feed-forward layers.
    
    Args:
        ff_prune_indices: Dictionary mapping layer names to indices to prune
        model: The SAINT model
        device: Computation device
        
    Returns:
        dict: Dictionary mapping layer names to binary masks
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pruning_masks = {}
    
    # First feed-forward network (column-wise)
    layer_name_1 = "ff_layer_col_output"
    if layer_name_1 in ff_prune_indices:
        ff_network_1 = model.transformer.layers[0][1].fn.fn.net
        last_linear_1 = ff_network_1[3]
        output_dim_1 = last_linear_1.weight.shape[0]  # Should be 32
        print("output_dim_1: ", output_dim_1)
        mask_1 = torch.ones(output_dim_1, device=device)
        mask_1[ff_prune_indices[layer_name_1]] = 0.0
        pruning_masks[layer_name_1] = mask_1
    
    # Second feed-forward network (row-wise)
    layer_name_2 = "ff_layer_row_output"
    if layer_name_2 in ff_prune_indices:
        ff_network_2 = model.transformer.layers[0][3].fn.fn.net
        last_linear_2 = ff_network_2[3]
        output_dim_2 = last_linear_2.weight.shape[0]  # Should be 1760
        print("output_dim_2: ", output_dim_2)
        mask_2 = torch.ones(output_dim_2, device=device)
        mask_2[ff_prune_indices[layer_name_2]] = 0.0
        pruning_masks[layer_name_2] = mask_2
    
    return pruning_masks

class PrunedSAINT(nn.Module):
    """
    A wrapper module that applies pruning masks to feed-forward outputs in SAINT.
    """
    def __init__(self, original_model, ff_pruning_masks):
        super().__init__()
        self.model = original_model
        self.ff_pruning_masks = ff_pruning_masks
        
        # Register hooks
        self.hooks = []
        self._register_pruning_hooks()
    
    def _register_pruning_hooks(self):
        """Register forward hooks to apply pruning masks"""
        # First feed-forward network (column-wise)
        layer_name_1 = "ff_layer_col_output"
        if layer_name_1 in self.ff_pruning_masks:
            ff_network_1 = self.model.transformer.layers[0][1].fn.fn.net
            last_linear_1 = ff_network_1[3]
            
            def get_pruning_hook(mask):
                def hook(module, input, output):
                    return output * mask
                return hook
            
            mask_1 = self.ff_pruning_masks[layer_name_1]
            hook_1 = last_linear_1.register_forward_hook(get_pruning_hook(mask_1))
            self.hooks.append(hook_1)
        
        # Second feed-forward network (row-wise)
        layer_name_2 = "ff_layer_row_output"
        if layer_name_2 in self.ff_pruning_masks:
            ff_network_2 = self.model.transformer.layers[0][3].fn.fn.net
            last_linear_2 = ff_network_2[3]
            
            mask_2 = self.ff_pruning_masks[layer_name_2]
            hook_2 = last_linear_2.register_forward_hook(get_pruning_hook(mask_2))
            self.hooks.append(hook_2)

    def forward(self, x):
        """Forward pass with pruning masks applied"""
        return self.model(x)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def prune_saint_layers(saint_model, dataset, prune_ratio):
    """
    Prunes feed-forward layers in SAINT.
    
    Args:
        saint_model: The SAINT model
        dataset: Dataset for collecting activations
        prune_ratio: Proportion of neurons to prune
        
    Returns:
        Pruned SAINT model
    """
    # 1. Collect feed-forward activations
    ff_activations = get_saint_ff_activations(saint_model, dataset)
    
    # 2. Analyze activations and get pruning indices
    ff_prune_indices = analyze_ff_activations(ff_activations, prune_ratio)
    
    # 3. Create pruning masks
    ff_pruning_masks = create_saint_pruning_masks(ff_prune_indices, saint_model.model)
    
    # 4. Create pruned model
    pruned_model = PrunedSAINT(saint_model.model, ff_pruning_masks)
    
    # 5. Replace the original model with the pruned one
    saint_model.model = pruned_model.model
    
    return saint_model




def prune_saint_y_reps(saint_model, dataset, prune_ratio, device=None):
    """
    Prunes the y_reps output in SAINT based on activation magnitudes.
    
    Args:
        saint_model: The SAINT model
        dataset: Dataset for collecting activations
        prune_ratio: Proportion of neurons to prune
        device: Computation device
        
    Returns:
        Pruned SAINT model with hooks applied to the CLS token output
    """
    batch_size = 64
    # 1. Collect y_reps activations

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    saint_model.to(device)
    saint_model.eval()

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Dictionary to store activations
    all_activations = []    
    # Collect activations
    with torch.no_grad():
        for batch in loader:
            # Process batch according to SAINT's input format
            inputs, targets = batch
            inputs = inputs.to(device)
            
            # Forward pass to get CLS token embeddings
            activations = saint_model.forward_embeddings(inputs)
            
            # Store activations
            all_activations.append(activations.cpu())

    # Concatenate activations from all batches
    all_activations = torch.cat(all_activations, dim=0)
    print(f"all_activations shape: {all_activations.shape}")
    
    # Calculate the mean absolute activation for each neuron
    mean_activations = torch.mean(torch.abs(all_activations), dim=0)
    print(f"mean_activations shape: {mean_activations.shape}")
    
    # Sort neurons by activation magnitude
    sorted_indices = torch.argsort(mean_activations)
    print(f"sorted_indices shape: {sorted_indices.shape}")
    
    # Select the bottom prune_ratio% of neurons (those with lowest activation)
    num_to_prune = int(prune_ratio * len(sorted_indices))
    indices_to_prune = sorted_indices[:num_to_prune].tolist()
    print(f"Pruning {num_to_prune}/{len(sorted_indices)} neurons from CLS token")
    
    # Create a mask for the CLS token output
    output_dim = mean_activations.shape[0]
    mask = torch.ones(output_dim, device=device, requires_grad=False)
    mask[indices_to_prune] = 0.0
    
    
    class PrunedMLPForY(nn.Module):
        def __init__(self, original_mlp_fory, mask):
            super().__init__()
            self.mlp_fory = original_mlp_fory
            self.mask = mask
            
        def forward(self, x):
            return self.mlp_fory(self.mask * x) 
        

    saint_model.model.mlpfory = PrunedMLPForY(saint_model.model.mlpfory, mask)

    return saint_model
            
            




##########################################################################################
########################## <EOS> End of SAINT Pruning Functions ##########################
##########################################################################################




def create_dataset_subset(dataset, subset_ratio=0.1, tuple_3=False):
    """
    Creates a subset of the given dataset with the specified ratio.
    
    Args:
        dataset (torch.utils.data.Dataset): The original dataset
        subset_ratio (float): Ratio of samples to keep (between 0 and 1)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        torch.utils.data.TensorDataset: A subset of the original dataset as a TensorDataset
    """
    import torch
    from torch.utils.data import TensorDataset, Subset
    import numpy as np
    
    # Calculate the number of samples to keep
    dataset_size = len(dataset)
    subset_size = int(dataset_size * subset_ratio)
    
    # Generate random indices without replacement
    indices = np.random.choice(dataset_size, subset_size, replace=False)
    
    # Create a temporary subset
    temp_subset = Subset(dataset, indices)
    
    

    if tuple_3:    
        all_data_c = []
        all_data_n = []
        all_labels = []
        for data_c, data_n, label in temp_subset:
            all_data_c.append(data_c)
            all_data_n.append(data_n)
            all_labels.append(label)
        
        # Convert to tensors
        data_tensor_c = torch.stack(all_data_c)
        data_tensor_n = torch.stack(all_data_n)
        label_tensor = torch.tensor(all_labels)
        
        # Create and return a TensorDataset
        return TensorDataset(data_tensor_c, data_tensor_n, label_tensor)
    else:
        # Extract data from the subset
        all_data = []
        all_labels = []
        for data, label in temp_subset:
            all_data.append(data)
            all_labels.append(label)
        
        # Convert to tensors
        data_tensor = torch.stack(all_data)
        label_tensor = torch.tensor(all_labels)
        
        # Create and return a TensorDataset
        return TensorDataset(data_tensor, label_tensor)

###############################################################################
# 5. Main function
###############################################################################
if __name__ == "__main__":
    """
    Fine-Pruning workflow:
      1) Load the poisoned model (FT-Transformer or SAINT).
      2) Collect feed-forward layer activations on a clean data subset.
      3) Identify and prune neurons with the lowest average activation.
      4) Evaluate the pruned model (CDA and ASR).
      5) Fine-tune the pruned model on a small clean data subset.
      6) Re-evaluate (CDA and ASR) and save results.
    """


    # set up an argument parser
    import argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--exp_num", type=int, default=0)
    parser.add_argument("--prune_rate", type=float, default=0.9)

    # parse the arguments
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "diabetes", "eye_movement", "kdd99", "credit_card", "covtype", "poker"]
    available_models = ["ftt", "tabnet", "saint"]

    # check and make sure that the dataset and model are available
    # the dataset and model names are case insensitive, so compare the lowercase versions of the arguments
    if args.dataset_name.lower() not in available_datasets:
        raise ValueError(f"Dataset {args.dataset_name} is not available. Please choose from: {available_datasets}")
    if args.model_name.lower() not in available_models:
        raise ValueError(f"Model {args.model_name} is not available. Please choose from: {available_models}")

     
     # check and make sure that mu, beta, lambd, and epsilon are within the valid range
    if args.mu < 0 or args.mu > 1:
        raise ValueError(f"Mu must be between 0 and 1. You provided: {args.mu}")
    if args.beta < 0 or args.beta > 1:
        raise ValueError(f"Beta must be between 0 and 1. You provided: {args.beta}")
    if args.lambd < 0 or args.lambd > 1:
        raise ValueError(f"Lambd must be between 0 and 1. You provided: {args.lambd}")
    if args.epsilon < 0 or args.epsilon > 1:
        raise ValueError(f"Epsilon must be between 0 and 1. You provided: {args.epsilon}")
    
    if args.prune_rate < 0 or args.prune_rate > 1:
        raise ValueError(f"Pruning rate must be between 0 and 1. You provided: {args.prune_rate}")
    

    # log all the arguments before running the experiment
    print("-"*100)
    logging.info(f"Pruning with the following arguments: {args}")
    print("-"*100)
    print("\n")

    model_name = args.model_name
    dataset_name = args.dataset_name
    target_label = args.target_label
    mu = args.mu
    beta = args.beta
    lambd = args.lambd
    epsilon = args.epsilon
    prune_rate = args.prune_rate


    dataset_dict = {
        "aci": ACI,
        "bm": BankMarketing,
        "higgs": HIGGS,
        "diabetes": Diabetes,
        "eye_movement": EyeMovement,
        "kdd99": KDD99,
        "credit_card": CreditCard,
        "covtype": CovType,
        "poker": Poker
    }

    model_dict = {
        "ftt": FTTModel,
        "tabnet": TabNetModel,
        "saint": SAINTModel,
        "catboost": CatBoostModel,
        "xgboost": XGBoostModel
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Step 1: Initialize the dataset object which can handle, convert and revert the dataset.
    data_obj = dataset_dict[dataset_name]()


    # create the experiment results directory
    results_path = Path("./results/pruning")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    csv_file_address = results_path / Path(f"{dataset_name}.csv")
    if not csv_file_address.exists():
        csv_file_address.touch()
        
        csv_header = ['EXP_NUM', 'DATASET', 'MODEL', 'TARGET_LABEL', 'EPSILON', 'PRUNE_RATE', 'MU', 'BETA', 'LAMBDA', 'P_CDA', 'P_ASR', 'FP_CDA', 'FP_ASR']
        # insert the header row into the csv file
        with open(csv_file_address, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)


    models_path = Path("./saved_models")

    # creating and checking the path for saving and loading the poisoned models.
    poisoned_model_path = models_path / Path(f"poisoned")
    if not poisoned_model_path.exists():
        poisoned_model_path.mkdir(parents=True, exist_ok=True)
    poisoned_model_address = poisoned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_poisoned_model.pth")


    # if the model name is xgboost, catboost, or tabnet, then we should add .zip at the end of the model name.
    if model_name in ['tabnet', 'xgboost', 'catboost']:
        # add .zip at the end of the model name in addition to the existing suffix
        poisoned_model_address_toload = poisoned_model_address.with_suffix(poisoned_model_address.suffix + ".zip")
    else:
        poisoned_model_address_toload = poisoned_model_address

    if not poisoned_model_address_toload.exists():
        raise ValueError(f"Poisoned model address at {poisoned_model_address_toload} does not exist.")
    


    
    if model_name == "ftt":
        model = FTTModel(data_obj=data_obj)
    elif model_name == "tabnet":
        model = TabNetModel(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type='entmax'
        )
    elif model_name == "saint":
        model = SAINTModel(data_obj=data_obj, is_numerical=False)

    if model_name == "ftt" and data_obj.cat_cols:
        model.load_model(poisoned_model_address_toload, model_type="original")  
        model.to(device, model_type="original")
    else:
        model.load_model(poisoned_model_address_toload)
        model.to(device)


    attack = Attack(device=device, model=model, data_obj=data_obj, target_label=target_label, mu=mu, beta=beta, lambd=lambd, epsilon=epsilon)

    attack.load_poisoned_dataset()

    poisoned_trainset, poisoned_testset = attack.poisoned_dataset
    poisoned_train_samples, poisoned_test_samples = attack.poisoned_samples
    attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
    attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)


    # Assuming poisoned_trainset and poisoned_train_samples are PyTorch datasets or tensors
    poisoned_indices = []


    # Convert poisoned_train_samples to a set for faster lookup
    poisoned_samples_set = set(tuple(sample.tolist()) for sample, _ in poisoned_train_samples)

    # Iterate over poisoned_trainset to find indices of poisoned samples
    for idx, (sample, _) in enumerate(poisoned_trainset):
        # Convert the sample to a tuple for comparison
        sample_tuple = tuple(sample.tolist())
        
        # Check if the sample is in the poisoned_samples_set
        if sample_tuple in poisoned_samples_set:
            poisoned_indices.append(idx)

    # poisoned_indices now contains the indices of poisoned samples in poisoned_trainset
    # print("Indices of poisoned samples:", poisoned_indices)

    # Step 11: Revert the poisoned dataset to the original categorical features
    FTT = True if attack.model.model_name == "FTTransformer" else False
    if data_obj.cat_cols:
        reverted_poisoned_trainset = attack.data_obj.Revert(attack.poisoned_dataset[0], FTT=FTT)
        reverted_poisoned_testset = attack.data_obj.Revert(attack.poisoned_dataset[1], FTT=FTT)
    else:
        reverted_poisoned_trainset = attack.poisoned_dataset[0]
        reverted_poisoned_testset = attack.poisoned_dataset[1]

    reverted_poisoned_dataset = (reverted_poisoned_trainset, reverted_poisoned_testset)

    # get the clean train and test datasets
    if FTT and data_obj.cat_cols:
        clean_dataset = attack.data_obj.get_normal_datasets_FTT()
    else:
        clean_dataset = attack.data_obj.get_normal_datasets()
    clean_trainset = clean_dataset[0]
    clean_testset = clean_dataset[1]


    tuple_3 = False
    
    if model_name == "ftt":

        model_type = "original" if data_obj.cat_cols else "converted"

        if data_obj.dataset_name == "covtype" or data_obj.dataset_name == "higgs" or data_obj.dataset_name == "poker":
            subset_ratio = 0.1
        else:
            subset_ratio = 1.0


        tuple_3 = True if model_type == "original" else False


        clean_subset = create_dataset_subset(clean_trainset, subset_ratio=subset_ratio, tuple_3=tuple_3)

        pruned_ftt_model =prune_ff_layers_example(model, clean_subset, prune_ratio=prune_rate, model_type=model_type)

        pruned_ftt_model.epochs = 5

    elif model_name == "saint":

        if data_obj.dataset_name == "covtype" or data_obj.dataset_name == "higgs" or data_obj.dataset_name == "poker":
            subset_ratio = 0.1
        else:
            subset_ratio = 1.0        

        clean_subset = create_dataset_subset(clean_trainset, subset_ratio=subset_ratio)

        model = prune_saint_layers(model, clean_subset, prune_ratio=prune_rate)
        

        pruned_saint_model = prune_saint_y_reps(model, clean_subset, prune_ratio=prune_rate)

        pruned_saint_model.opt.epochs = 5

        for name, param in pruned_saint_model.model.named_parameters():
            print(name, param.shape)

        print(pruned_saint_model.model.transformer)


    else:
        raise ValueError(f"Model {model_name} is not supported for pruning.")
    




    # test the pruned model on clean data and poisoned data
    converted = False if FTT and data_obj.cat_cols else True    
    p_asr = attack.test(reverted_poisoned_testset, converted=converted)
    p_cda = attack.test(clean_testset, converted=converted)

    print("==> [Step 4] Fine-Tuning the pruned model on clean data...")
    #Train the model on the poisoned training dataset
    
    attack.train((create_dataset_subset(clean_trainset, subset_ratio=0.05, tuple_3=tuple_3), clean_testset), converted=converted)
    logging.info("=== Fine-Tuning Completed ===")

    print("==> [Step 5] Testing the pruned model on clean data and poisoned data...")
    fp_asr = attack.test(reverted_poisoned_testset, converted=converted)
    fp_cda = attack.test(clean_testset, converted=converted)
    print("=== Testing Completed ===")



    # save the results to the csv file
    with open(csv_file_address, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([args.exp_num, dataset_name, model_name, target_label, epsilon, prune_rate, mu, beta, lambd, p_cda, p_asr, fp_cda, fp_asr])


    # creating and checking the path for saving and loading the pruned models.
    pruned_model_path = models_path / Path(f"pruned")
    if not pruned_model_path.exists():
        pruned_model_path.mkdir(parents=True, exist_ok=True)
    pruned_model_address = pruned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_pruned_model.pth")

    # Save the poisoned model with Unix timestamp in the filename
    if model_name == "ftt" and data_obj.cat_cols:
        attack.model.save_model(pruned_model_address, model_type="original")
    else:
        attack.model.save_model(pruned_model_address)

