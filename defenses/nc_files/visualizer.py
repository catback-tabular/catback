#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Metadata:
#   Date    : 2025-02-18
#   Author  : Behrad Tajalli
#   Description:
#       This file defines a Visualizer class for reverse-engineering backdoor 
#       triggers on tabular data. Unlike the image-domain version, here the 
#       input is a one-dimensional feature vector. Both the mask and the 
#       pattern are 1D arrays (of length equal to the number of features).
#
#       For the pattern, each feature has its own valid range provided by 
#       feature_min and feature_max. The transformation maps the pattern from 
#       its original scale to a normalized [0, 1] range before applying a tanh 
#       transform for unconstrained optimization.
#
#       The adversarial example is computed as:
#           X_adv = (1 - mask) * X + mask * pattern
#       where X is the original tabular input.
#
#       Loss is computed as the sum of a cross-entropy term (to enforce misclassification)
#       and a regularization term (to keep the mask small).
# ------------------------------------------------------------------------------

from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from decimal import Decimal

# Optionally, you can select a specific CUDA device.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Visualizer:
    """
    Visualizer for Tabular Data:
    
    This class implements the reverse-engineering optimization to recover a potential
    backdoor trigger for tabular data. The trigger consists of a pattern and a mask,
    each represented as a 1D vector (one row of features). The mask indicates which 
    features should be modified and the pattern specifies the new feature values.
    
    The original feature ranges (min and max for each feature) are provided so that the 
    pattern can be appropriately normalized and denormalized.
    """

    # -----------------------------
    # Class-level default parameters
    # -----------------------------
    # For the mask, we assume values in [0, 1].
    MASK_MIN = 0.0
    MASK_MAX = 1.0

    # Early stopping and dynamic cost adjustment settings.
    ATTACK_SUCC_THRESHOLD = 0.99  # Required attack success rate.
    PATIENCE = 10                 # Number of mini-batches to wait before cost adjustment.
    COST_MULTIPLIER = 1.5         # Multiplier for dynamic cost adjustment.
    RESET_COST_TO_ZERO = True     # Whether to reset the cost to zero at optimization start.
    
    VERBOSE = 1                   # Verbosity level.
    RETURN_LOGS = True            # Whether to return optimization logs.
    SAVE_LAST = False             # Whether to save the last pattern if no improvement.
    
    EPSILON = 1e-07               # Small epsilon for numerical stability in tanh.
    
    EARLY_STOP = True             # Enable early stopping.
    EARLY_STOP_THRESHOLD = 0.99   # Early stopping threshold.
    EARLY_STOP_PATIENCE = 2 * PATIENCE  # Early stopping patience.
    
    # Batch size for optimization.
    BATCH_SIZE = 32

    # Device selection.
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')
    
    def __init__(self, model, intensity_range, regularization, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 feature_min, feature_max,
                 attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 verbose=VERBOSE, return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 device=DEVICE):
        """
        Initialize the Visualizer for tabular data.

        Args:
            model: The tabular model to inspect.
            intensity_range (str): Preprocessing intensity range.
                                   For tabular data, use 'raw' (no scaling) as default.
            regularization (str): Regularization type ('l1' or 'l2').
            input_shape (tuple): Shape of input data (num_features,).
            init_cost (float): Initial cost weight to balance losses.
            steps (int): Total number of optimization iterations.
            mini_batch (int): Number of mini-batches per optimization step.
            lr (float): Learning rate for optimization.
            num_classes (int): Number of classes in the model.
            feature_min (array-like): 1D array of minimum valid values for each feature.
            feature_max (array-like): 1D array of maximum valid values for each feature.
            attack_succ_threshold (float): Target attack success rate.
            patience (int): Patience for dynamic cost adjustment.
            cost_multiplier (float): Multiplier for adjusting cost.
            reset_cost_to_zero (bool): Whether to reset cost to zero at start.
            mask_min (float): Minimum mask value (default 0.0).
            mask_max (float): Maximum mask value (default 1.0).
            verbose (int): Verbosity level.
            return_logs (bool): Whether to return logs of optimization.
            save_last (bool): Whether to save the final result if no improvement.
            epsilon (float): Small epsilon for numerical stability in tanh.
            early_stop (bool): Whether to enable early stopping.
            early_stop_threshold (float): Early stopping threshold.
            early_stop_patience (int): Patience for early stopping.
            device: Torch device (CPU or GPU).
        """
        # For tabular data, we use 'raw' intensity range.
        assert intensity_range in {'raw'}
        assert regularization in {None, 'l1', 'l2'}

        self.model = model
        self.intensity_range = intensity_range
        self.regularization = regularization
        # For tabular data, input_shape is (num_features,).
        self.input_shape = input_shape  
        self.init_cost = init_cost
        self.steps = steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes

        # Save feature-wise minimum and maximum values (convert to numpy arrays).
        self.feature_min = np.array(feature_min)  # shape: (num_features,)
        self.feature_max = np.array(feature_max)  # shape: (num_features,)

        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero

        self.mask_min = mask_min
        self.mask_max = mask_max

        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience

        self.device = device

        # Number of features (for convenience).
        self.num_features = input_shape[0]

        # (Optional) You could preallocate additional tensors here.
        # For tabular data, no upsampling is needed.

        self.raw_input_flag = True

    # -----------------------------
    # Reset Optimizer (if needed)
    # -----------------------------
    def reset_opt(self):
        """
        Reset optimizer state if required.
        In this implementation, we simply pass.
        """
        pass

    # -----------------------------
    # Reset Optimization State
    # -----------------------------
    def reset_state(self, pattern_init, mask_init):
        """
        Reset internal state before optimization.

        This involves:
          - Setting the balancing cost.
          - Clipping and converting the initial mask and pattern into tanh-space.
          - Preparing tensors for optimization.
        
        Args:
            pattern_init (np.array): Initial trigger pattern as a 1D vector (shape: [num_features]).
            mask_init (np.array): Initial mask as a 1D vector (shape: [num_features]).
        """
        print('resetting state')

        # Set cost: either reset to zero or use the initial cost.
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost

        self.cost_tensor = torch.from_numpy(np.array(self.cost)).to(self.device)

        # For the mask, assume values should lie in [mask_min, mask_max].
        mask = np.clip(mask_init, self.mask_min, self.mask_max)
        # For tabular, the mask is a 1D vector. We add a batch dimension.
        mask = np.expand_dims(mask, axis=0)  # shape: (1, num_features)

        # For the pattern, the valid range is feature_min to feature_max.
        pattern = np.clip(pattern_init, self.feature_min, self.feature_max)
        # Add batch dimension.
        pattern = np.expand_dims(pattern, axis=0)  # shape: (1, num_features)

        # --- Convert mask to tanh-space ---
        # For the mask, first normalize to [0, 1] (assumed to be already in [mask_min, mask_max],
        # and if mask_min=0 and mask_max=1 then it is already normalized).
        # Then transform: x_tanh = arctanh((x - 0.5)*(2-epsilon)).
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))

        # --- Convert pattern to tanh-space ---
        # First, normalize pattern: value_normalized = (pattern - feature_min) / (feature_max - feature_min)
        # This brings it to [0, 1] per feature.
        pattern_norm = (pattern - self.feature_min) / (self.feature_max - self.feature_min)
        pattern_tanh = np.arctanh((pattern_norm - 0.5) * (2 - self.epsilon))

        # Convert numpy arrays to torch tensors.
        self.mask_tanh_tensor = torch.Tensor(mask_tanh).to(self.device)  # shape: (1, num_features)
        self.mask_tanh_tensor.requires_grad = True

        self.pattern_tanh_tensor = torch.Tensor(pattern_tanh).to(self.device)  # shape: (1, num_features)
        self.pattern_tanh_tensor.requires_grad = True

        # Reset optimizer state.
        self.reset_opt()
        pass

    # -----------------------------
    # (Optional) Save Temporary Results
    # -----------------------------
    def save_tmp_func(self, step):
        """
        Optionally, save temporary results for debugging.
        For tabular data, you might simply save the current mask and pattern
        as CSV files or print them.
        
        Args:
            step (int): Current optimization step.
        """
        # Example: Save current mask (in raw space) to a text file.
        current_mask = torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5
        current_mask = current_mask.detach().cpu().numpy().squeeze()  # shape: (num_features,)
        filename = os.path.join('tmp', f'tmp_mask_step_{step}.csv')
        np.savetxt(filename, current_mask, delimiter=",")
        # Similarly, you can save the pattern.
        current_pattern_norm = torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5
        # Denormalize the pattern: pattern = pattern_norm * (max-min) + min
        current_pattern = current_pattern_norm * (self.feature_max - self.feature_min) + self.feature_min
        current_pattern = current_pattern.detach().cpu().numpy().squeeze()
        filename = os.path.join('tmp', f'tmp_pattern_step_{step}.csv')
        np.savetxt(filename, current_pattern, delimiter=",")
        pass

    # -----------------------------
    # Main Optimization Loop: visualize()
    # -----------------------------
    def visualize(self, gen, y_target, pattern_init, mask_init, ftt):
        """
        Run the optimization to recover a trigger pattern and mask for tabular data.
        
        Args:
            gen: Data generator (e.g., a DataLoader) that yields batches of tabular data.
                 Each batch should be of shape (batch_size, num_features).
            y_target (int): The target class label for which the trigger should force misclassification.
            pattern_init (np.array): Initial pattern as a 1D vector (num_features,).
            mask_init (np.array): Initial mask as a 1D vector (num_features,).
            ftt (bool): Whether the model is FTT or not. If True, instead of two: x-batch and y-batch, the dataloader will return 3 columns: x_cat, x_num, y
        Returns:
            If self.return_logs is True:
                (pattern_best, mask_best, mask_raw, logs)
            Else:
                (pattern_best, mask_best, mask_raw)
            
            Where:
              - pattern_best: The optimized trigger pattern (in original feature space).
              - mask_best: The optimized mask (as a 1D vector in [0,1]).
              - mask_raw: The mask in raw space (same as mask_best here).
              - logs: A log of the optimization process.
        """
        # Reset internal state.
        self.reset_state(pattern_init, mask_init)

        # Variables to hold the best result.
        mask_best = None
        pattern_best = None
        reg_best = float('inf')  # Best (lowest) regularization loss seen.

        # Logs for tracking progress.
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # --- Helper functions (for compatibility) ---
        def keras_preprocess(x_input, intensity_range):
            # For tabular data with 'raw', no preprocessing is applied.
            if intensity_range == 'raw':
                return x_input
            else:
                raise Exception('unknown intensity_range %s' % intensity_range)

        def keras_reverse_preprocess(x_input, intensity_range):
            # For 'raw', the input is unchanged.
            if intensity_range == 'raw':
                return x_input
            else:
                raise Exception('unknown intensity_range %s' % intensity_range)

        # Early stopping counters.
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        # Initialize the optimizer for the mask and pattern parameters.
        self.opt = optim.Adam([self.mask_tanh_tensor, self.pattern_tanh_tensor],
                              lr=self.lr, betas=[0.5, 0.9])
        # Use cross-entropy loss.
        ce_loss = torch.nn.CrossEntropyLoss()

        # Create a target tensor of shape (batch_size,) filled with y_target.
        Y_target = torch.from_numpy(np.array([y_target] * self.BATCH_SIZE)).long().to(self.device)

        # -----------------------------
        # Begin Optimization Loop
        # -----------------------------


        self.model.eval()

        loader_iter = cycle(gen)

        for step in range(self.steps):
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            used_samples = 0

            

            # Iterate over mini-batches.
            for idx in range(self.mini_batch):
                if ftt:
                    X_cat_batch, X_num_batch, Y_batch = next(loader_iter)  # Expect X_batch of shape (batch_size, num_features)
                    # Now, we should concatenate X_cat_batch and X_num_batch and remember the index which we concatenated them, so we can split them later
                    X_batch = torch.cat((X_cat_batch, X_num_batch), dim=1)
                else:
                    X_batch, Y_batch = next(loader_iter)  # Expect X_batch of shape (batch_size, num_features)
                # Reverse preprocessing if needed (here, for 'raw', nothing happens).
                if self.raw_input_flag:
                    input_raw_tensor = X_batch
                else:
                    input_raw_tensor = keras_reverse_preprocess(X_batch, self.intensity_range)
                input_raw_tensor = input_raw_tensor.to(self.device)

                # Recover mask in raw space from tanh representation.
                mask_raw = torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5  # shape: (1, num_features)
                self.mask_raw = mask_raw  # Save for later use.

                # Recover pattern in normalized [0,1] space.
                pattern_norm = torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5  # shape: (1, num_features)
                # Denormalize the pattern to original feature space.
                pattern_raw = pattern_norm * (torch.tensor(self.feature_max, device=self.device) - 
                                              torch.tensor(self.feature_min, device=self.device)) \
                                              + torch.tensor(self.feature_min, device=self.device)
                self.pattern_raw = pattern_raw

                # Create adversarial (triggered) input:
                # X_adv = (1 - mask) * X + mask * pattern.
                X_adv = (1 - mask_raw) * input_raw_tensor + mask_raw * pattern_raw
                # Ensure X_adv is on the correct device.
                X_adv = X_adv.to(self.device)

                # Adjust Y_target if batch size differs.
                if X_batch.shape[0] != Y_target.shape[0]:
                    Y_target = torch.from_numpy(np.array([y_target] * X_batch.shape[0])).long().to(self.device)

                # Forward pass through the model.
                if ftt:
                    output_tensor = self.model.forward_original(X_adv[:, :X_cat_batch.shape[1]].long(), X_adv[:, X_cat_batch.shape[1]:].float())
                else:
                    output_tensor = self.model.forward(X_adv)
                # Compute softmax and predicted labels.
                if output_tensor.device != self.device:
                    output_tensor = output_tensor.to(self.device)
                y_pred = F.softmax(output_tensor, dim=1)
                indices = torch.argmax(y_pred, 1)
                correct = torch.eq(indices, Y_target)
                loss_acc = torch.sum(correct).cpu().detach().item()
                loss_acc_list.append(loss_acc)
                used_samples += X_batch.shape[0]

                # Cross-entropy loss.
                loss_ce = ce_loss(output_tensor, Y_target)
                loss_ce_list.append(loss_ce.cpu().detach().item())

                # Regularization loss: L1 norm of the mask (averaged over features).
                loss_reg = torch.sum(torch.abs(mask_raw)) / self.num_features
                loss_reg = loss_reg.to(self.device)
                loss_reg_list.append(loss_reg.item())

                # Total loss: classification loss + cost * regularization.
                self.cost_tensor = self.cost_tensor.to(self.device)
                loss = loss_ce + loss_reg * self.cost_tensor
                loss_list.append(loss.cpu().detach().numpy())

                # Backpropagation.
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            # Compute average losses and attack success rate.
            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_loss_acc = np.sum(loss_acc_list) / used_samples

            # Save best result if attack success rate is met and regularization improves.
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = mask_raw.data.cpu().numpy().squeeze()  # shape: (num_features,)
                pattern_best = pattern_raw.data.cpu().numpy().squeeze()  # shape: (num_features,)
                reg_best = avg_loss_reg

            # Verbose logging.
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(self.cost), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            logs.append((step, avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc, reg_best, self.cost))

            # --- Early stopping check ---
            if self.early_stop:
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)
                if (cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # --- Dynamic cost adjustment ---
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    self.cost_tensor = torch.tensor(self.cost)
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost), Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                self.cost_tensor = torch.tensor(self.cost)
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost), Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                self.cost_tensor = torch.tensor(self.cost)
                cost_down_flag = True

            # Optionally, save temporary results.
            # Uncomment if debugging is desired.
            # if self.save_tmp:
            #     self.save_tmp_func(step)

        # If no best mask was found or if SAVE_LAST is set, use the final optimization result.
        if mask_best is None or self.save_last:
            mask_best = torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5
            mask_best = mask_best.data.cpu().numpy().squeeze()
            pattern_norm = torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5
            pattern_best = (pattern_norm * (torch.tensor(self.feature_max, device=self.device) - 
                                            torch.tensor(self.feature_min, device=self.device)) +
                            torch.tensor(self.feature_min, device=self.device))
            pattern_best = pattern_best.data.cpu().numpy().squeeze()

        if self.return_logs:
            return pattern_best, mask_best, logs
        else:
            return pattern_best, mask_best
