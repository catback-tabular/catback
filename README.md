# CatBack: Universal Backdoor Attacks on Tabular Data via Categorical Encoding

[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)](https://shields.io/)
[![Artifact Available](https://img.shields.io/badge/Artifact-Available-blue)](https://shields.io/)
[![Functional](https://img.shields.io/badge/Functional-Yes-green)](https://shields.io/)

This repository is prepared for the artifact evaluation of our accepted paper: **CatBack: Universal Backdoor Attacks on Tabular Data via Categorical Encoding**.

## Abstract

Backdoor attacks in machine learning have drawn significant attention for their potential to compromise models stealthily, yet most research has focused on homogeneous data such as images. In this work, we propose a novel backdoor attack on tabular data, which is particularly challenging due to the presence of both numerical and categorical features. 
Our key idea is a novel technique to convert categorical values into floating-point representations. This approach preserves enough information to maintain clean-model accuracy compared to traditional methods like one-hot or ordinal encoding. By doing this, we create a gradient-based universal perturbation that applies to all features, including categorical ones.

We evaluate our method on five benchmark datasets and four popular models. Our results show up to a 100% attack success rate in both white-box and black-box settings (including real-world applications like Vertex AI), revealing a severe vulnerability for tabular data. Our method is shown to surpass the previous works like Tabdoor in terms of performance, while remaining stealthy against state-of-the-art defense mechanisms. We evaluate our attack against Spectral Signatures, Neural Cleanse, Beatrix, and Fine-Pruning, all of which fail to defend successfully against it. We also verify that our attack successfully bypasses popular outlier detection mechanisms. 

## Overview

This repository implements the CatBack backdoor attack on tabular datasets. It includes dataset loaders, model implementations, and the attack logic. The main entry point is `step_by_step.py`, which performs the attack step by step.

## Project Structure

```
.
├── src
│   ├── attack.py                # Core attack logic
│   ├── dataset
│   │   ├── ACI.py               # Adult Census Income dataset
│   │   ├── BM.py                # Bank Marketing dataset
│   │   ├── CovType.py           # Covertype dataset
│   │   ├── CreditCard.py        # Credit Card Fraud dataset
│   │   ├── HIGGS.py             # HIGGS dataset
│   │   ├── __init__.py
│   │   └── Poker.py             # Poker Hand dataset
│   ├── __init__.py
│   ├── models
│   │   ├── FTT.py               # FT-Transformer model
│   │   ├── __init__.py
│   │   ├── saint_lib            # SAINT model dependencies
│   │   ├── SAINT.py             # SAINT model
│   │   ├── Tabnet.py            # TabNet model
│   │   └── XGBoost.py           # XGBoost model
│   └── __pycache__              # (Ignore: compiled Python files)
└── step_by_step.py              # Main script to run the attack

```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/catback-tabular/catback.git
   cd catback
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

Note: This repository requires Python 3.8+ and has been tested on Linux.

The versions in `requirements.txt` are pinned to match the development environment for reproducibility. If you encounter issues installing exact versions (e.g., due to OS or Python version incompatibilities), you can install the latest stable versions by removing the `==version` part from the file or using `pip install <package>` without versions.

## Datasets

Most datasets need to be manually downloaded and placed in the `./data/` directory. Create this directory if it doesn't exist. ACI and CovType datasets are automatically fetched and do not require manual download.

### ACI (Adult Census Income)
- Automatically downloaded via `shap.datasets.adult()` when running the code.

### CovType (Covertype)
- Automatically downloaded via `sklearn.datasets.fetch_covtype()` when running the code.

### BM (Bank Marketing)
- Download from [Kaggle: Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset).
- Place `bank.csv` in `./data/`.

### CreditCard (Credit Card Fraud)
- Download from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).
- Place `creditcard.csv` in `./data/`.

### HIGGS
- Download `HIGGS.csv.gz` from [UCI Machine Learning Repository: HIGGS](https://archive.ics.uci.edu/ml/datasets/HIGGS).
- Extract `HIGGS.csv`.
- Run the preprocessing script (HIGGS-preprocess.py, if available) to generate `processed.pkl` (the preprocessing file is provided in https://github.com/bartpleiter/tabular-backdoors as well).
- Place `processed.pkl` in `./data/`.

### Poker (Poker Hand)
- Download from [Kaggle: Poker Game Dataset](https://www.kaggle.com/datasets/hosseinah1/poker-game-dataset/data).
- Place `poker-hand-training.csv` and `poker-hand-testing.csv` in `./data/`.

Ensure file names match exactly as referenced in the code or you can modify the loading addresses in each dataset file if prefered.

## Usage

The attack is executed via the `step_by_step.py` script. It performs the following steps:
- Loads the dataset and model
- Trains a clean model (or loads if available)
- Generates and optimizes the backdoor trigger
- Poisons the dataset
- Trains the model on poisoned data
- Evaluates Clean Data Accuracy (CDA) and Attack Success Rate (ASR)

### Command-Line Arguments

- `--dataset_name` (required): Dataset to use (e.g., "aci", "bm", "higgs", "credit_card", "covtype", "poker")
- `--model_name` (required): Model to use (e.g., "ftt", "tabnet", "saint", "xgboost")
- `--target_label` (int, default=1): Target label for the backdoor
- `--mu` (float, default=0.2): Fraction of non-target samples to select
- `--beta` (float, default=0.1): L1 regularization hyperparameter
- `--lambd` (float, default=0.1): L2 regularization hyperparameter
- `--epsilon` (float, default=0.02): Poisoning rate for training data
- `--exp_num` (int, default=0): Experiment number for logging

### Example

Run the attack on the ACI dataset using FT-Transformer:

```
python step_by_step.py --dataset_name aci --model_name ftt --target_label 1 --epsilon 0.02
```

Results will be saved in `./results/<dataset_name>.csv`.

<!-- ## Reproducibility

To reproduce the results:
1. Install dependencies from `requirements.txt`.
2. Run the script with the desired parameters.
3. Models are saved in `./saved_models/` for reuse. -->


## Contact
For any questions or issues, please contact the authors:

- Behrad Tajalli: hamidreza.tajalli@ru.nl
- Stefanos Koffas: S.Koffas@tudelft.nl
- Stjepan Picek: stjepan.picek@ru.nl


## License

[MIT License](LICENSE) 