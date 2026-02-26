# ss.py
"""
Spectral Signature Defense
===============================================================

This file contains:

1) Minimal dataset and DataLoader wrappers.
2) Feature-extraction stubs for TabNet, SAINT, and FT-Transformer.
3) The spectral_signature_defense() function.
4) The plotCorrelationScores() function for visualizing score distributions.
5) An example usage section at the bottom illustrating how to tie it all together.

Note:
 - You must adapt the "extract_features_*" functions to your actual model implementations.
 - Where you see 'model.forward_embeddings()', you can replace these calls with the correct logic or hooks
   for your own codebase.
"""

from pathlib import Path
import sys
import os

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






# ============================================
# Logging Configuration
# ============================================

# Set up the logging configuration to display informational messages.
# This helps in tracking the progress and debugging if necessary.
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the logging message format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)



###############################################################################
# 2. Feature-Extraction Stubs
###############################################################################
def extract_features_tabnet(
    tabnet_model,
    dataset,
    batch_size: int = 128,
    device: torch.device = None
) -> np.ndarray:
    """
    Extract penultimate-layer (or suitable hidden) features from TabNet.

    Steps:
      1) Move model to `device`, put in eval mode.
      2) Create a DataLoader from X,y.
      3) For each batch, run a method that returns the hidden embedding
         (for instance, model.forward_embeddings(...)).
      4) Collect and return all embeddings as a single (N, D) array.

    IMPORTANT:
     - You must implement or reference the actual function/forward hook
       that obtains hidden embeddings from your TabNet model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tabnet_model.to(device)
    tabnet_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            # batch_x = batch_x.float().to(device)
            # Hypothetical call that returns penultimate-layer features:
            # Replace with your actual approach (forward hook or direct method).
            features = tabnet_model.forward_embeddings(batch_x)  # <-- implement in your code
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def extract_features_saint(
    saint_model,
    dataset,
    batch_size: int = 128,
    device: torch.device = None
) -> np.ndarray:
    """
    Extract penultimate embeddings from a trained SAINT model in batch.

    You must define how exactly you retrieve the hidden representation
    (e.g., using forward_saint_features(...) or a forward hook).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saint_model.to(device)
    saint_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            # batch_x = batch_x.float().to(device)
            # Acquire penultimate-layer features:
            penult_feats = saint_model.forward_embeddings(batch_x)
            all_features.append(penult_feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)



def extract_features_fttransformer(
    ft_model,
    dataset,
    data_obj,
    batch_size: int = 128,
    device: torch.device = None
) -> np.ndarray:
    """
    Batched penultimate feature extraction for FT-Transformer.
    X_c: categorical features
    X_n: continuous features
    y: labels
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft_model.to(device, model_type="original")
    ft_model.eval()



    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        if data_obj.cat_cols:
            for batch_c, batch_n, _ in loader:
                batch_c = batch_c.to(device)
                batch_n = batch_n.to(device)
                feats = ft_model.forward_embeddings(batch_c, batch_n)
                all_features.append(feats.cpu().numpy())
        else:
            for batch_n, _ in loader:
                batch_n = batch_n.to(device)
                batch_c = torch.empty(batch_n.shape[0], 0, dtype=torch.long)
                batch_c = batch_c.to(device)
                feats = ft_model.forward_embeddings(batch_c, batch_n)
                all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


###############################################################################
# 3. Spectral Signature Defense
###############################################################################
def spectral_signature_defense(
    features: np.ndarray,
    labels: np.ndarray,
    class_list,
    expected_poison_fraction: float = 0.05,
    extra_multiplier: float = 1.5,
):
    """
    Implements the Spectral Signature defense on extracted features.

    For each class c:
      1) Gather features of that class.
      2) Center them.
      3) Perform SVD -> top singular vector.
      4) Compute squared projection along that vector as score.
      5) Remove top K samples (K ~ expected_poison_fraction * extra_multiplier).
    
    Returns:
      - suspicious_indices (List[int]): indices flagged as suspicious
      - scores_by_class (Dict[int, np.ndarray]): arrays of scores for each class
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must match in length.")

    suspicious_indices = []
    scores_by_class = {}
    all_indices = np.arange(len(labels))

    for c in class_list:
        # Indices for class c
        class_idxs = all_indices[labels == c]
        if len(class_idxs) < 2:
            # Not enough samples to do meaningful SVD
            continue

        class_feats = features[class_idxs]
        mean_feat = class_feats.mean(axis=0)
        centered = class_feats - mean_feat

        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        top_vec = Vt[0]

        # Score = squared projection
        scores = (centered @ top_vec) ** 2
        scores_by_class[c] = scores

        # Number of suspicious to remove
        K = int(len(class_feats) * expected_poison_fraction * extra_multiplier)
        K = min(K, len(class_feats))
        if K < 1:
            continue

        # The top-K scoring samples are suspicious
        suspicious_local = np.argsort(scores)[-K:]  # largest scores
        suspicious_global = class_idxs[suspicious_local]
        suspicious_indices.extend(suspicious_global)

    return suspicious_indices, scores_by_class


###############################################################################
# 4. Plot Correlation Scores
###############################################################################
# def plotCorrelationScores(
#     class_id: int,
#     scores_for_class: np.ndarray,
#     mask_for_class: np.ndarray,
#     nbins: int = 100,
#     label_clean: str = "Clean",
#     label_poison: str = "Poisoned",
#     save_path: Path = None
# ):
#     """
#     Plots a histogram of the correlation (spectral) scores for clean vs. poisoned
#     or suspicious samples in a single class.

#     Args:
#       - class_id (int): The class label for the plot title
#       - scores_for_class (np.ndarray): Array of shape (N_class_samples,) with the spectral scores
#       - mask_for_class (np.ndarray): Boolean array of same shape, True means "poisoned" or "suspicious"
#       - nbins (int): Number of bins for the histogram
#       - label_clean (str): Legend label for clean distribution
#       - label_poison (str): Legend label for poison distribution
#     """
#     plt.figure(figsize=(5,3))
#     sns.set_style("white")
#     sns.set_palette("tab10")

#     scores_clean = scores_for_class[~mask_for_class]
#     scores_poison = scores_for_class[mask_for_class]

#     if len(scores_poison) == 0:
#         # No poison => just plot clean
#         if len(scores_clean) == 0:
#             return  # no data
#         bins = np.linspace(0, scores_clean.max(), nbins)
#         plt.hist(scores_clean, bins=bins, color="green", alpha=0.75, label=label_clean)
#     else:
#         # We have both categories
#         combined_max = max(scores_clean.max(), scores_poison.max())
#         bins = np.linspace(0, combined_max, nbins)
#         plt.hist(scores_clean, bins=bins, color="green", alpha=0.75, label=label_clean)
#         plt.hist(scores_poison, bins=bins, color="red", alpha=0.75, label=label_poison)
#         plt.legend(loc="upper right")

#     plt.xlabel("Spectral Signature Score")
#     plt.ylabel("Count")
#     plt.title(f"Class {class_id}: Score Distribution")
#     plt.show()
#     # save the plot
#     if save_path is not None:
#         plt.savefig(save_path)
#     else:
#         plt.savefig(f"plots/ss_class_{class_id}.png")


###############################################################################
# 4. Plot Correlation Scores
###############################################################################
def plotCorrelationScores(
    class_id: int,
    scores_for_class: np.ndarray,
    mask_for_class: np.ndarray,
    nbins: int = 100,
    label_clean: str = "Clean",
    label_poison: str = "Poisoned",
    save_path: Path = None,
    figsize=(4, 3),
    dpi: int = 300,
):
    """
    Plot compact, publication-friendly histograms of spectral scores for one class.

    Design goals for paper figures:
      - Small physical size with high DPI for crisp text/lines.
      - Minimal surrounding whitespace when saved (bbox_inches='tight').
      - In-axes legend with reduced font to avoid expanding the bounding box.

    Args:
      - class_id: Class label for title.
      - scores_for_class: (N_class,) spectral scores.
      - mask_for_class: Boolean mask identifying poisoned/suspicious samples.
      - nbins: Histogram bin count.
      - label_clean/label_poison: Legend labels.
      - save_path: Optional path to save the figure.
      - figsize: Tuple width x height in inches for compact figure.
      - dpi: Dots-per-inch for sharp rendering in print/PDF.
    """
    # Configure a compact figure footprint with crisp rendering
    plt.figure(figsize=figsize, dpi=dpi)
    sns.set_style("white")
    sns.set_palette("tab10")

    scores_clean = scores_for_class[~mask_for_class]
    scores_poison = scores_for_class[mask_for_class]

    # Compute shared bins so distributions are directly comparable
    if len(scores_poison) == 0:
        if len(scores_clean) == 0:
            return  # no data to plot
        bins = np.linspace(0, scores_clean.max(), nbins)
        plt.hist(scores_clean, bins=bins, color="green", alpha=0.8, label=label_clean)
    else:
        combined_max = max(scores_clean.max(), scores_poison.max())
        bins = np.linspace(0, combined_max, nbins)
        plt.hist(scores_clean, bins=bins, color="green", alpha=0.8, label=label_clean)
        plt.hist(scores_poison, bins=bins, color="red", alpha=0.75, label=label_poison)
        # Keep legend inside axes with slightly larger font for readability
        plt.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)

    # Compact labeling suitable for limited space in papers
    plt.xlabel("Spectral Signature Score", fontsize=9)
    plt.ylabel("Count", fontsize=9)
    plt.title(f"Class {class_id}: Score Distribution", fontsize=12)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=8, length=3)
    # Reduce default margins to reclaim space inside the axes
    plt.margins(x=0.01)
    # Tight layout reduces internal padding; we'll also use bbox_inches='tight' on save
    plt.tight_layout(pad=0.2)

    # Save first to ensure tight bounding box is applied to the file
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(f"plots/ss_class_{class_id}.png", bbox_inches='tight', pad_inches=0.02)

    # Optional on-screen preview for interactive runs
    plt.show()
    plt.close()

###############################################################################
# 5. Main function
###############################################################################
if __name__ == "__main__":
    """
    In practice, you would:
      1) Train or load your model (TabNet, SAINT, or FT-Transformer).
      2) Extract penultimate features via the appropriate function.
      3) Run spectral_signature_defense(...) to identify suspicious samples.
      4) If you have a ground-truth poison mask, or if you want to treat
         suspicious indices as "poison", you can plot with plotCorrelationScores.
    """


    # set up an argument parser
    import argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.2)
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


    if model_name == "ftt":
        features = extract_features_fttransformer(model, reverted_poisoned_trainset, data_obj, batch_size=128, device=device)
    elif model_name == "tabnet":
        features = extract_features_tabnet(model, reverted_poisoned_trainset, batch_size=128, device=device)
    elif model_name == "saint":
        features = extract_features_saint(model, reverted_poisoned_trainset, batch_size=128, device=device)

    # the last tensor in reverted_poisoned_trainset is the labels, we need to convert it to numpy array
    labels = reverted_poisoned_trainset.tensors[-1].cpu().numpy()

    class_list = [num for num in range(data_obj.num_classes)]

    suspicious_idx, scores_dict = spectral_signature_defense(
        features,
        labels=labels,
        class_list=class_list,
        expected_poison_fraction=epsilon,
        extra_multiplier=1.5
    )

    poison_mask = np.zeros(len(labels), dtype=bool)
    poison_mask[poisoned_indices] = True

    save_path = Path("./plots")
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # Now we can plot the distribution for each class
    for c in class_list:
        # The scores for class c are in scores_dict[c].
        # The associated indices are where y_demo == c.
        class_idxs = np.where(labels == c)[0]
        class_scores = scores_dict[c]
        class_poison_mask = poison_mask[class_idxs]
        save_address = save_path / Path(f"ss_class{c}_{model_name}_{dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}.png")

        plotCorrelationScores(
            class_id=c,
            scores_for_class=class_scores,
            mask_for_class=class_poison_mask,
            nbins=50,
            label_clean="Clean",
            label_poison="Poisoned",
            save_path=save_address
        )

    print("Done with Spectral Signature demo!")
    
    


    # # Let's create a small synthetic example for demonstration:
    # N = 1000
    # D = 20
    # X_demo = np.random.randn(N, D).astype(np.float32)
    # y_demo = np.random.randint(0, 3, size=N)  # 3 classes: 0,1,2



    # # Suppose we have "features" from a model. For demonstration, let's pretend
    # # they are just X_demo. In reality, you call extract_features_*.
    # features_demo = X_demo.copy()  # placeholder

    # # Let's run spectral signature
    # class_list = [0,1,2]
    # suspicious_idx, scores_dict = spectral_signature_defense(
    #     features_demo,
    #     labels=y_demo,
    #     class_list=class_list,
    #     expected_poison_fraction=0.1,
    #     extra_multiplier=1.5
    # )

    # # Let's assume we have a "poisoned_mask" if we know which are truly poison.
    # # For demonstration, create a random mask of 30 "poison" samples.
    # poison_mask = np.zeros(N, dtype=bool)
    # poison_indices = np.random.choice(N, size=100, replace=False)
    # poison_mask[poison_indices] = True

    # # Now we can plot the distribution for each class
    # for c in class_list:
    #     # The scores for class c are in scores_dict[c].
    #     # The associated indices are where y_demo == c.
    #     class_idxs = np.where(y_demo == c)[0]
    #     class_scores = scores_dict[c]
    #     class_poison_mask = poison_mask[class_idxs]

    #     plotCorrelationScores(
    #         class_id=c,
    #         scores_for_class=class_scores,
    #         mask_for_class=class_poison_mask,
    #         nbins=50,
    #         label_clean="Clean",
    #         label_poison="Poisoned"
    #     )

    # print("Done with Spectral Signature demo!")
