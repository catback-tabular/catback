# Beatrix_tabular.py
"""
Beatrix Defense (Tabular Adaptation)
===============================================================

This file implements the Beatrix (BEAT) backdoor detection defense
adapted for tabular data. The original Beatrix defense was designed
for image classifiers; this version operates on penultimate-layer
embeddings extracted from tabular models (FT-Transformer, SAINT, TabNet).

The detection works by:
  1) Extracting feature representations from the model.
  2) Computing Gram-matrix-based feature correlations per class.
  3) Using MAD-based thresholds to identify out-of-distribution samples.
  4) Measuring KMMD distance between clean and suspicious feature groups.
  5) Flagging classes with anomalously high KMMD scores as backdoored.

Note:
 - You must adapt model and dataset paths to your actual setup.
"""

import csv
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset



from pathlib import Path
import sys
import os

# Add the project root to the Python path
# Get the absolute path of the current file
current_file = Path(__file__).resolve()
# Get the project root (two directories up from the current file)
project_root = current_file.parent.parent
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











# =============================================================================
# Gaussian Kernel and KMMD Distance Functions
# =============================================================================
def gaussian_kernel(x1, x2, kernel_mul=2.0, kernel_num=5, fix_sigma=0, mean_sigma=0):
    """
    Compute a multi-scale Gaussian kernel between two sets of samples.
    
    Args:
        x1 (torch.Tensor): First set of samples.
        x2 (torch.Tensor): Second set of samples.
        kernel_mul (float): Multiplicative factor for adjusting the bandwidth.
        kernel_num (int): Number of kernels to sum.
        fix_sigma (float): Fixed sigma value (if provided).
        mean_sigma (float): If provided, uses mean L2 distance as sigma.
        
    Returns:
        torch.Tensor: The summed Gaussian kernel values.
    """
    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]
    x1_tile_shape = []
    x2_tile_shape = []
    norm_shape = []
    for i in range(len(x1.shape) + 1):
        if i == 1:
            x1_tile_shape.append(x2_sample_size)
        else:
            x1_tile_shape.append(1)
        if i == 0:
            x2_tile_shape.append(x1_sample_size)
        else:
            x2_tile_shape.append(1)
        if i not in [0, 1]:
            norm_shape.append(i)
    tile_x1 = torch.unsqueeze(x1, 1).repeat(x1_tile_shape)
    tile_x2 = torch.unsqueeze(x2, 0).repeat(x2_tile_shape)
    L2_distance = torch.square(tile_x1 - tile_x2).sum(dim=norm_shape)
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:
        bandwidth = torch.median(L2_distance.reshape(L2_distance.shape[0], -1))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def kmmd_dist(x1, x2):
    """
    Compute the Kernel Maximum Mean Discrepancy (KMMD) distance between two sets of features.
    
    Args:
        x1 (torch.Tensor): First set of features.
        x2 (torch.Tensor): Second set of features.
        
    Returns:
        float: The computed KMMD distance.
    """
    X_total = torch.cat([x1, x2], 0)
    Gram_matrix = gaussian_kernel(X_total, X_total, kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    x1x1 = Gram_matrix[:n, :n]
    x2x2 = Gram_matrix[n:, n:]
    x1x2 = Gram_matrix[:n, n:]
    diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
    diff = (m * n) / (m + n) * diff
    return diff.to(torch.device('cpu')).numpy()

# =============================================================================
# Feature Correlations Class for Tabular Data
# =============================================================================
class Feature_Correlations:
    def __init__(self, POWER_list, mode='mad'):
        """
        Initialize the Feature_Correlations detector.
        
        Args:
            POWER_list (list): List of powers (orders) for Gram matrix computation.
            mode (str): Method for deviation calculation (default: 'mad').
        """
        self.power = POWER_list
        self.mode = mode

    def train(self, in_data):
        """
        Train the detector using a set of feature tensors.
        
        Args:
            in_data (list): List containing feature tensors (each of shape [batch, feature_dim]).
        """
        self.in_data = in_data
        if 'mad' in self.mode:
            self.medians, self.mads = self.get_median_mad(self.in_data)
            self.mins, self.maxs = self.minmax_mad()

    def minmax_mad(self):
        """
        Compute minimum and maximum thresholds using median ± 10 * MAD.
        
        Returns:
            tuple: Two lists (mins, maxs) for each feature order.
        """
        mins = []
        maxs = []
        for L, (medians, mads) in enumerate(zip(self.medians, self.mads)):
            if L == len(mins):
                mins.append([None] * len(self.power))
                maxs.append([None] * len(self.power))
            for p in range(len(self.power)):
                mins[L][p] = medians[p] - mads[p] * 10
                maxs[L][p] = medians[p] + mads[p] * 10
        return mins, maxs

    def G_p(self, ob, p):
        """
        Compute the power-normalized Gram matrix for the given features.
        
        Args:
            ob (torch.Tensor): Feature tensor.
            p (int or float): The power to use.
            
        Returns:
            torch.Tensor: Flattened Gram matrix features.
        """
        temp = ob.detach() ** p
        temp = temp.reshape(temp.shape[0], -1)
        temp = torch.matmul(temp.unsqueeze(2), temp.unsqueeze(1))
        # Only consider the upper triangular part
        temp = torch.triu(temp)
        temp = temp.sign() * torch.abs(temp) ** (1 / p)
        temp = temp.reshape(temp.shape[0], -1)
        self.num_feature = temp.shape[-1] / 2  # because of symmetry
        return temp

    def get_median_mad(self, feat_list):
        """
        Compute medians and MAD for each power order.
        
        Args:
            feat_list (list): List of feature tensors.
            
        Returns:
            tuple: Lists of medians and MADs.
        """
        medians = []
        mads = []
        for L, feat_L in enumerate(feat_list):
            if L == len(medians):
                medians.append([None] * len(self.power))
                mads.append([None] * len(self.power))
            for p in range(len(self.power)):
                g_p = self.G_p(feat_L, self.power[p])
                current_median = g_p.median(dim=0, keepdim=True)[0]
                current_mad = torch.abs(g_p - current_median).median(dim=0, keepdim=True)[0]
                medians[L][p] = current_median
                mads[L][p] = current_mad
        return medians, mads

    def get_deviations_(self, feat_list):
        """
        Compute deviation scores using thresholding on Gram matrix values.
        
        Args:
            feat_list (list): List of feature tensors.
            
        Returns:
            np.array: Deviation scores per sample.
        """
        deviations = []
        batch_deviations = []
        for L, feat_L in enumerate(feat_list):
            dev = 0
            for p in range(len(self.power)):
                g_p = self.G_p(feat_L, self.power[p])
                dev += (F.relu(self.mins[L][p] - g_p) / torch.abs(self.mins[L][p] + 1e-6)).sum(dim=1, keepdim=True)
                dev += (F.relu(g_p - self.maxs[L][p]) / torch.abs(self.maxs[L][p] + 1e-6)).sum(dim=1, keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations, axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations, axis=0) / self.num_feature / len(self.power)
        return deviations

    def get_deviations(self, feat_list):
        """
        Compute deviation scores as normalized absolute differences from medians.
        
        Args:
            feat_list (list): List of feature tensors.
            
        Returns:
            np.array: Normalized deviation scores.
        """
        deviations = []
        batch_deviations = []
        for L, feat_L in enumerate(feat_list):
            dev = 0
            for p in range(len(self.power)):
                g_p = self.G_p(feat_L, self.power[p])
                dev += torch.sum(torch.abs(g_p - self.medians[L][p]) / (self.mads[L][p] + 1e-6), dim=1, keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations, axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations, axis=0) / self.num_feature / len(self.power)
        return deviations

# =============================================================================
# Threshold Determination Function
# =============================================================================
def threshold_determine(clean_feature_target, ood_detection):
    """
    Determine the 95th and 99th percentile thresholds for deviation scores.
    
    Args:
        clean_feature_target (torch.Tensor): Features for a given class.
        ood_detection (Feature_Correlations): The OOD detector instance.
        
    Returns:
        tuple: (percentile_95, percentile_99)
    """
    test_deviations_list = []
    step = 5
    for i in range(step):
        index_mask = np.ones((len(clean_feature_target),))
        idx_start = i * int(len(clean_feature_target) // step)
        idx_end = (i + 1) * int(len(clean_feature_target) // step)
        index_mask[idx_start:idx_end] = 0
        train_feats = clean_feature_target[np.where(index_mask == 1)]
        test_feats = clean_feature_target[np.where(index_mask == 0)]
        ood_detection.train(in_data=[train_feats])
        test_deviations = ood_detection.get_deviations_([test_feats])
        test_deviations_list.append(test_deviations)
    test_deviations = np.concatenate(test_deviations_list, axis=0)
    test_deviations_sort = np.sort(test_deviations, axis=0)
    percentile_95 = test_deviations_sort[int(len(test_deviations_sort) * 0.95)][0]
    percentile_99 = test_deviations_sort[int(len(test_deviations_sort) * 0.99)][0]
    print(f'percentile_95: {percentile_95}')
    print(f'percentile_99: {percentile_99}')
    return percentile_95, percentile_99

# =============================================================================
# BEAT Detector Class for Tabular Data
# =============================================================================
class BEAT_detector:
    def __init__(self, target_label, clean_test=500, bd_test=500, order_list=np.arange(1, 9)):
        """
        Initialize the BEAT detector.
        
        Args:
            opt: Configuration options (should include target_label, num_classes, etc.).
            clean_test (int): Number of clean samples to test per class.
            bd_test (int): Number of poisoned samples to test for target class.
            order_list (np.array): Orders for Gram matrix computation.
        """
        self.test_target_label = target_label
        self.order_list = order_list
        # For tabular data, you may choose a fixed number of samples per class.
        self.clean_data_perclass = 30
        self.clean_test = clean_test
        self.bd_test = bd_test

    def _detecting(self, data, target_label, num_classes, device):
        """
        Run the BEAT detection procedure on the extracted features.
        
        Args:
            data (dict): Dictionary with keys:
                - 'feature': Tensor of penultimate embeddings (y_reps) for all samples.
                - 'labels': Tensor of corresponding predicted (or true) labels.
        """
        
        # Here, we assume that 'feature' is a tensor of shape [N, D]
        features = data['feature'].to(device)
        labels = data['labels'].cpu().numpy()

        # Shuffle the features and labels
        features, labels = shuffle(features, labels)

        # Instantiate the OOD detector
        ood_detection = Feature_Correlations(POWER_list=self.order_list, mode='mad')
        J_t = []  # List to store KMMD distance for each class
        threshold_list = []

        # Loop through each class (assume classes from 0 to num_classes-1)
        for test_target_label in range(num_classes):
            print(f'***** Class: {test_target_label} *****')
            # Extract features for the current class (based on labels)
            class_indices = np.where(labels == test_target_label)[0]
            if len(class_indices) == 0:
                continue
            clean_feature_target = features[class_indices]
            # For threshold determination, use a subset of samples for this class
            clean_feature_defend = clean_feature_target[:self.clean_data_perclass]
            threshold_95, threshold_99 = threshold_determine(clean_feature_defend, ood_detection)
            threshold_list.append([test_target_label, threshold_95, threshold_99])

            ood_detection.train(in_data=[clean_feature_defend])
            # Use the last clean_test samples from this class as test features
            clean_feature_test = features[class_indices][-self.clean_test:]
            # Label 0 for clean, and 1 for poisoned (only for the target class)
            clean_label_test = np.zeros((clean_feature_test.shape[0],))
            
            # For the target backdoor class, assume that some samples are poisoned.
            # Here, we simply use features from the same class as "poisoned" for demonstration.
            if test_target_label == target_label:
                # In practice, you might have separate information to extract known poisoned samples.
                bd_feature_test = clean_feature_target[:self.bd_test]
                bd_label_test = np.ones((bd_feature_test.shape[0],))
                feature_test = torch.cat([clean_feature_test, bd_feature_test], 0)
                label_test = np.concatenate([clean_label_test, bd_label_test], 0)
                # For analysis, compute deviation percentiles between clean and poison
                clean_deviations_sort = np.sort(ood_detection.get_deviations_([clean_feature_test]), axis=0)
                bd_deviations_sort = np.sort(ood_detection.get_deviations_([bd_feature_test]), axis=0)
                p95 = clean_deviations_sort[int(len(clean_deviations_sort) * 0.95)][0]
                percentile_95 = np.where(bd_deviations_sort > p95, 1, 0)
                print(f'Class {test_target_label} percentile_95: {p95}, TP95: {percentile_95.sum()/len(bd_deviations_sort)}')
            else:
                feature_test = clean_feature_test
                label_test = clean_label_test

            test_deviations = ood_detection.get_deviations_([feature_test])
            ood_label_95 = np.where(test_deviations > threshold_95, 1, 0).squeeze()
            false_negative = np.where(label_test - ood_label_95 > 0, 1, 0).sum()
            false_positive = np.where(label_test - ood_label_95 < 0, 1, 0).sum()
            print(f'False negatives: {false_negative}, False positives: {false_positive}')

            # For grouping, split features based on the ood label
            clean_group = feature_test[np.where(ood_label_95 == 0)]
            bd_group = feature_test[np.where(ood_label_95 == 1)]
            # Flatten groups by taking mean (if features are multi-dimensional; for tabular, they may be 2D already)
            if len(clean_group.shape) > 2:
                clean_flat = torch.mean(clean_group, dim=1)
            else:
                clean_flat = clean_group
            if len(bd_group.shape) > 2:
                bd_flat = torch.mean(bd_group, dim=1)
            else:
                bd_flat = bd_group
            if bd_flat.shape[0] < 1:
                kmmd = np.array([0.0])
            else:
                kmmd = kmmd_dist(clean_flat, bd_flat)
            print(f'KMMD for class {test_target_label}: {kmmd.item() if hasattr(kmmd, "item") else kmmd}')
            J_t.append(kmmd.item() if hasattr(kmmd, "item") else kmmd)
        
        print("KMMD distances per class:", J_t)
        J_t = np.asarray(J_t)
        # Standardize the distances using median and MAD
        J_t_median = np.median(J_t)
        J_MAD = np.median(np.abs(J_t - J_t_median))
        J_star = np.abs(J_t - J_t_median) / (1.4826 * (J_MAD + 1e-6))
        for i, score in enumerate(J_star):
            print(f'Class {i}: J_star = {score:.2f}')
        # Save result to file
        # self._save_result_to_dir(result=[J_star])


        # return the J_star and kmmd value for target class
        return J_star[target_label], J_t[target_label]

    def _save_result_to_dir(self, result):
        """
        Save detection results to a file.
        
        Args:
            result (list): List containing the standardized scores.
        """
        opt = self.opt
        result_dir = os.path.join(opt.result, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, opt.attack_mode, f'target_{opt.target_label}')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        output_path = os.path.join(result_path, f"{opt.attack_mode}_{opt.dataset}_output.txt")
        with open(output_path, "w+") as f:
            result_str = ", ".join(str(v) for v in result[0])
            f.write(result_str + "\n")
        print(f"Results saved to {output_path}")

# =============================================================================
# Feature Extraction from the Poisoned Dataset Using SAINT
# =============================================================================
def extract_features(model_obj, dataloader, ftt_3, device):
    """
    Extract penultimate embeddings (y_reps) and predicted labels from the SAINT model.
    
    Args:
        model_obj (SAINTModel): Instance of the SAINTModel.
        dataloader (DataLoader): DataLoader for the poisoned dataset.
        device: Device to run the model on.
        
    Returns:
        dict: Dictionary with keys:
              'feature': tensor of extracted embeddings,
              'labels': tensor of predicted labels.
    """
    features_list = []
    labels_list = []

    if model_obj.model_name == "SAINT": 
    
        model_obj.eval()
        for batch in dataloader:
            # Assume batch is a tuple (inputs, target_labels)
            inputs, targets = batch
            inputs = inputs.to(device)
            # Get penultimate embeddings using forward_embeddings
            reps = model_obj.forward_embeddings(inputs)
            features_list.append(reps.detach().cpu())
            # Also obtain predictions (you can use model_obj.forward for final outputs)
            outputs = model_obj.forward(inputs)
            preds = torch.argmax(outputs, dim=1)
            labels_list.append(preds.detach().cpu())
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return {"feature": features, "labels": labels}
    
    elif model_obj.model_name == "FTTransformer": 
        model_type = "original" if ftt_3 else "converted"
        model_obj.to(device, model_type=model_type)
        for batch in dataloader:
            if ftt_3: 
                X_c, X_n, targets = batch
                X_c = X_c.to(device)
                X_n = X_n.to(device)
                cls_reps = model_obj.forward_clstokens(X_c, X_n, model_type)
                features_list.append(cls_reps.detach().cpu())
                outputs = model_obj.forward_original(X_c, X_n)
                preds = torch.argmax(outputs, dim=1)
                labels_list.append(preds.detach().cpu())
            else:
                X_n, targets = batch
                X_n = X_n.to(device)
                X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long).to(device)
                cls_reps = model_obj.forward_clstokens(X_c, X_n, model_type)
                features_list.append(cls_reps.detach().cpu())
                outputs = model_obj.forward(X_n)
                preds = torch.argmax(outputs, dim=1)
                labels_list.append(preds.detach().cpu())
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return {"feature": features, "labels": labels}
    
    elif model_obj.model_name == "TabNet":
        model_obj.eval()
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            outputs = model_obj.forward(inputs)
            embeddings = model_obj.forward_embeddings(inputs)
            preds = torch.argmax(outputs, dim=1)
            features_list.append(embeddings.detach().cpu())
            labels_list.append(preds.detach().cpu())
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return {"feature": features, "labels": labels}
    else: 
        raise ValueError(f"Model {model_obj.model_name} is not supported.")
            
            





###############################################################################
# 5. Main function
###############################################################################
if __name__ == "__main__":

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

    # parse the arguments
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "diabetes", "eye_movement", "kdd99", "credit_card", "covtype"]
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
    

    # log all the arguments before running the experiment
    print("-"*100)
    logging.info(f"Defending with the following arguments: {args}")
    print("-"*100)
    print("\n")

    model_name = args.model_name
    dataset_name = args.dataset_name
    target_label = args.target_label
    mu = args.mu
    beta = args.beta
    lambd = args.lambd
    epsilon = args.epsilon


    dataset_dict = {
        "aci": ACI,
        "bm": BankMarketing,
        "higgs": HIGGS,
        "diabetes": Diabetes,
        "eye_movement": EyeMovement,
        "kdd99": KDD99,
        "credit_card": CreditCard,
        "covtype": CovType
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





    # create the experiment results directory
    results_path = Path("./results/beatrix")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    csv_file_address = results_path / Path(f"{dataset_name}.csv")
    if not csv_file_address.exists():
        csv_file_address.touch()
        
        csv_header = ['EXP_NUM', 'DATASET', 'MODEL', 'TARGET_LABEL', 'EPSILON', 'ATTACK_TYPE', 'MU', 'BETA', 'LAMBDA', 'TRIGGER_SIZE', 'KMMD', 'J_STAR']
        # insert the header row into the csv file
        with open(csv_file_address, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    attack_type = "catback"
    trigger_size = 0






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
    ftt_3 = None
    if FTT and data_obj.cat_cols:
        ftt_3 = True
        clean_dataset = attack.data_obj.get_normal_datasets_FTT()
    else:
        clean_dataset = attack.data_obj.get_normal_datasets()
        ftt_3 = False
    clean_trainset = clean_dataset[0]
    clean_testset = clean_dataset[1]


    # Create a DataLoader for the poisoned train dataset
    dataloader = DataLoader(reverted_poisoned_trainset, batch_size=32, shuffle=False)

    # Extract features and predicted labels from the dataset.
    feature_data = extract_features(model, dataloader, ftt_3, device)

    print(feature_data["feature"].shape)
    print(feature_data["labels"].shape)

    # Initialize and run the BEAT detector.
    beat_detector = BEAT_detector(target_label=target_label)
    j_star, kmmd = beat_detector._detecting(feature_data, target_label, data_obj.num_classes, device)
    

    # save the results to the csv file
    with open(csv_file_address, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([args.exp_num, dataset_name, model_name, target_label, epsilon, attack_type, mu, beta, lambd, trigger_size, kmmd, j_star])

