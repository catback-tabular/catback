# CatBack: Universal Backdoor Attacks on Tabular Data via Categorical Encoding

<!-- [![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)](https://shields.io/)
[![Artifact Available](https://img.shields.io/badge/Artifact-Available-blue)](https://shields.io/)
[![Functional](https://img.shields.io/badge/Functional-Yes-green)](https://shields.io/) -->

This repository is prepared for the artifact evaluation of our accepted paper: **CatBack: Universal Backdoor Attacks on Tabular Data via Categorical Encoding**.

## Abstract

This paper introduces CatBack, a novel backdoor attack specifically designed for tabular data containing both numerical and categorical features. The main research theme addresses a significant gap in backdoor attack research, which has predominantly focused on homogeneous data like images, leaving tabular data vulnerabilities largely unexplored.

**Research Contribution**: Our key innovation is a novel categorical encoding technique that converts categorical values into floating-point representations while preserving sufficient information to maintain clean model accuracy. This enables the creation of gradient-based universal perturbations that work across all feature types, including categorical ones.

**Artifact Support**: This repository provides complete implementation artifacts that demonstrate:
1. **Novel Categorical Encoding**: Implementation of our frequency-based hierarchical mapping technique for converting categorical features to numerical representations
2. **Universal Backdoor Attack**: Complete attack pipeline that generates universal triggers applicable to all features
3. **Comprehensive Evaluation**: Support for 5 benchmark datasets (ACI, Bank Marketing, CoverType, Credit Card, HIGGS, Poker) and 4 popular models (FT-Transformer, TabNet, SAINT, XGBoost)
4. **Performance Validation**: Reproducible experiments showing up to 100% attack success rates in both white-box and black-box settings

**Expected Workflow**: The experimental workflow follows these steps:
1. **Dataset Preparation**: Load and preprocess tabular datasets with mixed feature types
2. **Clean Model Training**: Train baseline models on original data to establish performance benchmarks
3. **Categorical Conversion**: Apply our novel encoding to transform categorical features into numerical representations
4. **Trigger Generation**: Optimize universal backdoor triggers using gradient-based methods
5. **Data Poisoning**: Inject backdoor samples into training data at specified poisoning rates
6. **Backdoor Training**: Retrain models on poisoned datasets
7. **Evaluation**: Measure Clean Data Accuracy (CDA) and Attack Success Rate (ASR) to demonstrate attack effectiveness

The artifacts enable researchers to compare against existing methods, and extend our approach to new datasets or models.

## Overview

This repository implements the CatBack backdoor attack on tabular datasets. It includes dataset loaders, model implementations, and the attack logic. The main entry point is `step_by_step.py`, which performs the attack step by step.

## Project Structure

```
.
├── src
│   ├── attack.py                # Core attack logic
│   ├── dataset/
│   │   ├── ACI.py               # Adult Census Income dataset
│   │   ├── BM.py                # Bank Marketing dataset
│   │   ├── CovType.py           # Covertype dataset
│   │   ├── CreditCard.py        # Credit Card Fraud dataset
│   │   ├── HIGGS.py             # HIGGS dataset
│   │   ├── __init__.py
│   │   └── Poker.py             # Poker Hand dataset
│   ├── __init__.py
│   ├── models/
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
   git clone <repository-url>
   cd catback
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
The versions in `requirements.txt` are pinned to match the development environment for reproducibility. If you encounter issues installing exact versions (e.g., due to OS or Python version incompatibilities), you can install the latest stable versions by removing the `==version` part from the file or using `pip install <package>` without versions.

## Hardware and Software Requirements

### Hardware Requirements
- **CPU**: Modern multi-core processor, e.g., Intel/AMD x86_64 - (tested on Intel Xeon Platinum 8360Y @ 2.40GHz)
- **RAM**: Minimum 8GB, recommended 16GB+ for larger datasets - (tested with 32GB available)
- **Storage**: At least 16GB free space for datasets and model storage
- **GPU**: Needed for faster training: CUDA-compatible GPU with 4GB+ VRAM - (tested on NVIDIA A100-SXM4-40GB)

### Software Requirements
- **Operating System**: Linux - (tested on RHEL 9.4)
- **Python**: Version 3.8 or higher - (tested with Python 3.11.3)
- **CUDA**: version 11.0+ for using GPU acceleration - (tested with CUDA 12.4)
- **PyTorch**: Compatible with CUDA version - (tested with PyTorch 2.5.1+cu124)

### Notes
- The artifact can run on commodity hardware (standard desktop/laptop)
- GPU acceleration is optional but significantly speeds up training

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
- Run the preprocessing script (HIGGS-preprocess.py available in `./data/`) to generate `processed.pkl` (the preprocessing file is provided in https://github.com/bartpleiter/tabular-backdoors as well).
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
- `--mu` (float, default=1.0): Fraction of non-target samples to select
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

## Contact
For any questions or issues, please contact the authors:

- Behrad Tajalli: hamidreza.tajalli@ru.nl
- Stefanos Koffas: S.Koffas@tudelft.nl
- Stjepan Picek: stjepan.picek@ru.nl

## License

[MIT License](LICENSE) 