#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Metadata:
#   Date    : 2025-02-18
#   Author  : Behrad
#   Description:
#       This script implements the reverse-engineering of potential backdoor triggers
#       for tabular data using the Neural Cleanse defense.
#
#       It loads a pre-trained tabular model, a tabular dataset (from a CSV file),
#       and iterates over each target label to optimize and recover a trigger.
#
#       The trigger consists of a pattern and a mask (both 1D vectors, one per sample).
#       The recovered trigger is saved as CSV files.
# ------------------------------------------------------------------------------

import os
import time
import random
import numpy as np
import torch


from torch.utils.data import DataLoader, TensorDataset
from .visualizer import Visualizer  # Our tabular version of Visualizer (see previous file)

# ------------------------------------------------------------------------------
# PARAMETERS (adjust these to your dataset and model)
# ------------------------------------------------------------------------------

# Device setup.
use_cuda = torch.cuda.is_available()
print("CUDA available:", use_cuda)
print("Torch version:", torch.__version__)
DEVICE = torch.device('cuda' if use_cuda else 'cpu')


# # Result saving.
# RESULT_DIR = 'results_tabular_nc'        # Directory to store the recovered trigger results.
# if not os.path.exists(RESULT_DIR):
#     os.mkdir(RESULT_DIR)
# Filename template for saving the recovered pattern and mask.
# For tabular data, we save CSV files.
FILE_FILENAME_TEMPLATE = 'tabular_visualize_%s_label_%d.csv'


# Optimization (reverse-engineering) parameters.
BATCH_SIZE = 32       # Batch size for optimization.
LR = 0.01              # Learning rate.
STEPS = 1000          # Total number of optimization iterations.
NB_SAMPLE = 1000      # Total samples used per mini-batch step.
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # Number of mini-batches per optimization iteration.
INIT_COST = 1e-3      # Initial cost weight for balancing objectives.
REGULARIZATION = 'l1' # Regularization type for controlling mask norm.

ATTACK_SUCC_THRESHOLD = 0.99  # Required attack success rate.
PATIENCE = 5                  # Patience for cost adjustment.
COST_MULTIPLIER = 2           # Multiplier for dynamic cost adjustment.
SAVE_LAST = False             # Whether to save the final result if no improvement is found.

# Early stopping parameters.
EARLY_STOP = True
EARLY_STOP_THRESHOLD = 1.0
EARLY_STOP_PATIENCE = 5 * PATIENCE



# ------------------------------------------------------------------------------
# Save Recovered Trigger Function for Tabular Data
# ------------------------------------------------------------------------------
def save_pattern(pattern, mask, y_target, results_dir):
    """
    Save the recovered trigger pattern and mask as CSV files.
    
    Args:
        pattern (np.array): Recovered trigger pattern (1D vector of length num_features).
        mask (np.array): Recovered mask (1D vector with values in [0,1]).
        y_target (int): The target label for which the trigger is recovered.
    """
    # Save the pattern.
    pattern_filename = os.path.join(results_dir, FILE_FILENAME_TEMPLATE % ('pattern', y_target))
    np.savetxt(pattern_filename, pattern, delimiter=",")
    # Save the mask.
    mask_filename = os.path.join(results_dir, FILE_FILENAME_TEMPLATE % ('mask', y_target))
    np.savetxt(mask_filename, mask, delimiter=",")
    # Create and save a fusion (element-wise product) as well.
    fusion = pattern * mask  # For tabular, fusion can be interpreted elementwise.
    fusion_filename = os.path.join(results_dir, FILE_FILENAME_TEMPLATE % ('fusion', y_target))
    np.savetxt(fusion_filename, fusion, delimiter=",")
    print(f"Saved pattern, mask, and fusion for label {y_target} to {results_dir}")
    pass

# ------------------------------------------------------------------------------
# Reverse-Engineering and Visualization Routine
# ------------------------------------------------------------------------------
def tabular_visualize_label_scan(results_dir, num_features, num_classes, feature_min, feature_max, model, dataloader, ftt):
    """
    Main routine to perform label scanning and reverse-engineering for tabular data.
    
    

    This function:
      - Loads the tabular dataset from a CSV file.
      - Loads the pre-trained tabular model.
      - Initializes the Visualizer (tabular version) with the appropriate hyperparameters.
      - Iterates over each target label (with Y_TARGET prioritized) and runs the
        reverse-engineering optimization.
      - Saves the recovered trigger pattern and mask for each label.
    """

    input_shape = (num_features,)  # Each sample is a 1D vector of features.
    y_target = 0                     # (Optional) Prioritized target label for backdoor injection.


    # Initialize the Visualizer for tabular data.
    visualizer = Visualizer(
        model,
        intensity_range='raw',  # For tabular data, we assume 'raw'
        regularization=REGULARIZATION,
        input_shape=input_shape,
        init_cost=INIT_COST,
        steps=STEPS,
        mini_batch=MINI_BATCH,
        lr=LR,
        num_classes=num_classes,
        feature_min=feature_min,
        feature_max=feature_max,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE,
        cost_multiplier=COST_MULTIPLIER,
        reset_cost_to_zero=SAVE_LAST,  # You may adjust this flag.
        verbose=2,
        return_logs=True,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP,
        early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE,
        device=DEVICE
    )

    log_mapping = {}

    # Create list of labels to scan. Prioritize Y_TARGET.
    y_target_list = list(range(num_classes))
    if y_target_list.count(y_target):
        y_target_list.remove(y_target)
        y_target_list = [y_target] + y_target_list

    for y_target in y_target_list:
        print('Processing label %d' % y_target)
        # Run the reverse-engineering optimization for the current label.
        pattern, mask_upsample, logs = visualizer.visualize(
            gen=dataloader,
            y_target=y_target,
            pattern_init=np.random.random(input_shape),  # Random initialization for pattern.
            mask_init=np.random.random(input_shape),        # Random initialization for mask.
            ftt=ftt
        )
        # Save the recovered trigger.
        save_pattern(pattern, mask_upsample, y_target, results_dir)
        log_mapping[y_target] = logs

    # Optionally, save log_mapping or further analyze logs.
    pass

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
# def main():
#     """
#     Main entry point for reverse-engineering the trigger on tabular data.
#     """
#     start_time = time.time()
#     tabular_visualize_label_scan()
#     elapsed_time = time.time() - start_time
#     print('Elapsed time: %.2f s' % elapsed_time)

# if __name__ == '__main__':
#     main()
