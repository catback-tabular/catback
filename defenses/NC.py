# NC.py
"""
Neural Cleanse Defense
===============================================================

This file implements the Neural Cleanse defense, which reverse-engineers
potential backdoor triggers by optimizing for minimal perturbation patterns
that cause misclassification to each target label. It then uses Median
Absolute Deviation (MAD) based outlier detection to identify anomalous
(potentially backdoored) labels.

Note:
 - Requires the 'nc_files' module which provides reverse_trigger and
   mad_outlier_detection utilities.
 - You must adapt the model and dataset paths to your actual setup.
"""

import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from attack import Attack
from nc_files import reverse_trigger, mad_outlier_detection



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




###############################################################################
# 5. Main function
###############################################################################
if __name__ == "__main__":
    """
    Neural Cleanse workflow:
      1) Load the (potentially poisoned) model.
      2) For each class, reverse-engineer a minimal trigger pattern using
         optimization (reverse_trigger.tabular_visualize_label_scan).
      3) Apply MAD-based outlier detection on the resulting trigger norms
         to identify anomalously small triggers (indicating a backdoor).
      4) Report the anomaly score for the target label.
    """


    # set up an argument parser
    import argparse
    parser = argparse.ArgumentParser()

    # add required arguments for all attack types
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--exp_num", type=int, default=0)
    parser.add_argument("--attack_type", type=str, default="catback", choices=["catback", "none", 'badnet', 'tabdoor'])

    # add conditional arguments based on attack type
    # these arguments are only needed for catback attack
    parser.add_argument("--mu", type=float, default=1.0, help="Only used when attack_type is catback")
    parser.add_argument("--beta", type=float, default=0.1, help="Only used when attack_type is catback")
    parser.add_argument("--lambd", type=float, default=0.1, help="Only used when attack_type is catback")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Used for catback, badnet, and tabdoor attacks")

    # add trigger_size argument for badnet attack
    parser.add_argument("--trigger_size", type=float, default=0.08, choices=[0.02, 0.08], help="Only used when attack_type is badnet")

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

    # Validate arguments based on attack type
    if args.attack_type == "catback":
        # check and make sure that mu, beta, lambd, and epsilon are within the valid range
        if args.mu < 0 or args.mu > 1:
            raise ValueError(f"Mu must be between 0 and 1. You provided: {args.mu}")
        if args.beta < 0 or args.beta > 1:
            raise ValueError(f"Beta must be between 0 and 1. You provided: {args.beta}")
        if args.lambd < 0 or args.lambd > 1:
            raise ValueError(f"Lambd must be between 0 and 1. You provided: {args.lambd}")
        if args.epsilon < 0 or args.epsilon > 1:
            raise ValueError(f"Epsilon must be between 0 and 1. You provided: {args.epsilon}")
    elif args.attack_type == "badnet":
        # Only validate epsilon and trigger_size for badnet
        if args.epsilon < 0 or args.epsilon > 1:
            raise ValueError(f"Epsilon must be between 0 and 1. You provided: {args.epsilon}")
        if args.trigger_size not in [0.02, 0.08]:
            raise ValueError(f"Trigger size must be either 0.02 or 0.08. You provided: {args.trigger_size}")
    elif args.attack_type == "tabdoor":
        # Only validate epsilon for tabdoor
        if args.epsilon < 0 or args.epsilon > 1:
            raise ValueError(f"Epsilon must be between 0 and 1. You provided: {args.epsilon}")
    # For "none" attack type, no additional validation is needed

    # log all the arguments before running the experiment
    print("-"*100)
    logging.info(f"Neural Cleanse with the following arguments: {args}")
    print("-"*100)
    print("\n")

    model_name = args.model_name
    dataset_name = args.dataset_name
    target_label = args.target_label
    attack_type = args.attack_type
    epsilon = args.epsilon
    
    # Set parameters based on attack type
    if attack_type == "catback":
        mu = args.mu
        beta = args.beta
        lambd = args.lambd
        trigger_size = 0.0
    elif attack_type == "badnet":
        trigger_size = args.trigger_size
        # Set default values for catback parameters since they're not used
        mu = 0.0
        beta = 0.0
        lambd = 0.0
    elif attack_type == 'tabdoor' or attack_type == 'none':  # tabdoor or none
        # Set default values for catback parameters since they're not used
        mu = 0.0
        beta = 0.0
        lambd = 0.0
        trigger_size = 0.0
    else:
        raise ValueError(f"Attack type {attack_type} is not supported.")

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

    if attack_type == "catback":

        models_path = Path("./saved_models")

        # creating and checking the path for saving and loading the poisoned models.
        poisoned_model_path = models_path / Path(f"poisoned")
        if not poisoned_model_path.exists():
            raise ValueError(f"Poisoned model path at {poisoned_model_path} does not exist.")
        poisoned_model_address = poisoned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_poisoned_model.pth")

    elif attack_type == "badnet":
        poisoned_model_address = Path("./saved_models/badnet/poisoned") / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{trigger_size}_{epsilon}_poisoned_model_bn.pth")

    elif attack_type == "tabdoor":
        poisoned_model_address = Path("./saved_models/tabdoor/poisoned") / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{epsilon}_poisoned_model_td.pth")

    elif attack_type == "none":
        if data_obj.dataset_name == "bank_marketing":
            poisoned_model_address = Path("./clean_saved_models/ordinal") / Path(f"{model_name}_bm.pth")
        else:
            poisoned_model_address = Path("./clean_saved_models/ordinal") / Path(f"{model_name}_{data_obj.dataset_name}.pth")

    else:
        raise ValueError(f"Attack type {attack_type} is not supported.")


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
        is_numerical = False
        model = SAINTModel(data_obj=data_obj, is_numerical=is_numerical)

    if model_name == "ftt" and data_obj.cat_cols:
        model.load_model(poisoned_model_address_toload, model_type="original")
        model.to(device, model_type="original")
    else:
        model.load_model(poisoned_model_address_toload)
        model.to(device)



    # create the experiment results directory
    results_path = Path("./results/nc")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    csv_file_address = results_path / Path(f"{dataset_name}.csv")
    if not csv_file_address.exists():
        csv_file_address.touch()
        
        csv_header = ['EXP_NUM', 'DATASET', 'MODEL', 'TARGET_LABEL', 'EPSILON', 'ATTACK_TYPE', 'MU', 'BETA', 'LAMBDA', 'TRIGGER_SIZE', 'ANOMALY_SCORE']
        # insert the header row into the csv file
        with open(csv_file_address, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
    




    attack = Attack(device=device, model=model, data_obj=data_obj, target_label=target_label, mu=mu, beta=beta, lambd=lambd, epsilon=epsilon)

    # attack.load_poisoned_dataset()

    # poisoned_trainset, poisoned_testset = attack.poisoned_dataset
    # poisoned_train_samples, poisoned_test_samples = attack.poisoned_samples
    # attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
    # attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)


    # # Assuming poisoned_trainset and poisoned_train_samples are PyTorch datasets or tensors
    # poisoned_indices = []


    # # Convert poisoned_train_samples to a set for faster lookup
    # poisoned_samples_set = set(tuple(sample.tolist()) for sample, _ in poisoned_train_samples)

    # # Iterate over poisoned_trainset to find indices of poisoned samples
    # for idx, (sample, _) in enumerate(poisoned_trainset):
    #     # Convert the sample to a tuple for comparison
    #     sample_tuple = tuple(sample.tolist())
        
    #     # Check if the sample is in the poisoned_samples_set
    #     if sample_tuple in poisoned_samples_set:
    #         poisoned_indices.append(idx)

    # poisoned_indices now contains the indices of poisoned samples in poisoned_trainset
    # print("Indices of poisoned samples:", poisoned_indices)

    # Step 11: Revert the poisoned dataset to the original categorical features
    FTT = True if attack.model.model_name == "FTTransformer" else False
    # if data_obj.cat_cols:
    #     reverted_poisoned_trainset = attack.data_obj.Revert(attack.poisoned_dataset[0], FTT=FTT)
    #     reverted_poisoned_testset = attack.data_obj.Revert(attack.poisoned_dataset[1], FTT=FTT)
    # else:
    #     reverted_poisoned_trainset = attack.poisoned_dataset[0]
    #     reverted_poisoned_testset = attack.poisoned_dataset[1]

    # reverted_poisoned_dataset = (reverted_poisoned_trainset, reverted_poisoned_testset)

    # get the clean train and test datasets
    if FTT and data_obj.cat_cols:
        is_ftt = True
        clean_dataset = attack.data_obj.get_normal_datasets_FTT()
    else:
        is_ftt = False
        clean_dataset = attack.data_obj.get_normal_datasets()
    clean_trainset = clean_dataset[0]
    clean_testset = clean_dataset[1]



    # Create a directory for saving the masks 
    masks_dir = Path("./saved_nc_masks")
        
    mask_address = masks_dir / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}")

    if not mask_address.exists():
        mask_address.mkdir(parents=True, exist_ok=True)


    # number of features (columns) in the dataset
    num_features = len(data_obj.feature_names)

    # Calculate the minimum and maximum values for each feature in the dataset
    features_min = data_obj.X_encoded.min().values.astype(np.float32)
    features_max = data_obj.X_encoded.max().values.astype(np.float32)

    dataloader = DataLoader(clean_testset, batch_size=32, shuffle=True)


    reverse_trigger.tabular_visualize_label_scan(results_dir=mask_address,
                                                num_features=num_features,
                                                num_classes=data_obj.num_classes,
                                                feature_min=features_min, 
                                                feature_max=features_max, 
                                                model=model, 
                                                dataloader=dataloader,
                                                ftt=is_ftt)
    

    anomaly_dict = mad_outlier_detection.analyze_pattern_norm_dist(results_dir=mask_address, num_classes=data_obj.num_classes)

    # if target label is not in the anomaly dictionary, then print the anomaly dictionary
    if target_label not in anomaly_dict:
        raise ValueError(f"Target label {target_label} is not in the anomaly dictionary: {anomaly_dict}")
    else:
        print(f"anomaly score for target label {target_label}: {anomaly_dict[target_label]}")

    # save the results to the csv file
    with open(csv_file_address, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([args.exp_num, dataset_name, model_name, target_label, epsilon, attack_type, mu, beta, lambd, trigger_size, anomaly_dict[target_label]])





