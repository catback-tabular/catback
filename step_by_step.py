
import torch
import logging
import time # Import time module for Unix timestamp
from pathlib import Path
import csv
import os

from src.attack import Attack

# importing all the models for this project
from src.models.FTT import FTTModel
from src.models.Tabnet import TabNetModel
from src.models.SAINT import SAINTModel
from src.models.XGBoost import XGBoostModel

# importing all the datasets for this project
from src.dataset.BM import BankMarketing
from src.dataset.ACI import ACI
from src.dataset.HIGGS import HIGGS
from src.dataset.CreditCard import CreditCard
from src.dataset.CovType import CovType  # Importing the ConvertCovType class from CovType.py
from src.dataset.Poker import Poker





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





# ============================================
# Main Execution Block
# ============================================


def attack_step_by_step(args, use_saved_models, use_existing_trigger):
    """
    The main function orchestrates the attack setup by executing the initial model training steps,
    retraining on the poisoned dataset, and evaluating the backdoor's effectiveness.
    """

    dataset_name = args.dataset_name
    model_name = args.model_name
    target_label = args.target_label
    mu = args.mu
    beta = args.beta
    lambd = args.lambd
    epsilon = args.epsilon
    exp_num = args.exp_num

    models_path = Path("./saved_models")

    # create the experiment results directory
    results_path = Path("./results")
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    csv_file_address = results_path / Path(f"{dataset_name}.csv")
    if not csv_file_address.exists():
        csv_file_address.touch()
        
        csv_header = ['EXP_NUM', 'DATASET', 'MODEL', 'TARGET_LABEL', 'EPSILON', 'MU', 'BETA', 'LAMBDA', 'BA_CONVERTED', 'CDA', 'ASR']
        # insert the header row into the csv file
        with open(csv_file_address, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
        



    dataset_dict = {
        "aci": ACI,
        "bm": BankMarketing,
        "higgs": HIGGS,
        "credit_card": CreditCard,
        "covtype": CovType,
        "poker": Poker
    }

    model_dict = {
        "ftt": FTTModel,
        "tabnet": TabNetModel,
        "saint": SAINTModel,
        "xgboost": XGBoostModel
    }




    logging.info("=== Starting Initial Model Training ===")

    # Step 0: Initialize device, target label, and other parameters needed for the attack.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Initialize the dataset object which can handle, convert and revert the dataset.
    data_obj = dataset_dict[dataset_name](args)

    # Step 2: Initialize the model. If needed (optional), the model can be loaded from a saved model. Then the model is not needed to be trained again.

    if model_name == "ftt":
        model = FTTModel(data_obj=data_obj, args=args)
    elif model_name == "tabnet" or model_name == "catboost" or model_name == "xgboost":
        model = TabNetModel(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            momentum=0.3,
            mask_type='entmax',
            args=args
        )
    elif model_name == "saint":
        model = SAINTModel(data_obj=data_obj, is_numerical=True, args=args)
    
    # creating and checking the path for saving and loading the clean models.
    clean_model_path = models_path / Path(f"converted_clean")
    if not clean_model_path.exists():
        clean_model_path.mkdir(parents=True, exist_ok=True)
    clean_model_address = clean_model_path / Path(f"{model_name}_{data_obj.dataset_name}.pth")

    # if the model name is xgboost or tabnet, then we should add .zip at the end of the model name.
    if model_name in ['tabnet', 'xgboost']:
        # add .zip at the end of the model name in addition to the existing suffix
        clean_model_address_toload = clean_model_address.with_suffix(clean_model_address.suffix + ".zip")
    else:
        clean_model_address_toload = clean_model_address

    # if the clean trained model is to be used, then load the model from the path.
    if use_saved_models['clean']: 
        if clean_model_address_toload.exists():
            logging.info(f"Loading the clean model from {clean_model_address_toload}")
            model.load_model(clean_model_address_toload)            
        else:
            logging.warning(f"Clean model address at {clean_model_address_toload} does not exist.")
            logging.warning("Training the model from scratch.")
            use_saved_models['clean'] = False


    if model_name == "ftt" and data_obj.cat_cols:
        model.to(device, model_type="original")
 
    model.to(device)


    # Step 3: Initialize the Attack class with model, data, and target label
    attack = Attack(device=device, model=model, data_obj=data_obj, target_label=target_label, mu=mu, beta=beta, lambd=lambd, epsilon=epsilon, args=args)

    # Step 4: Train the model on the clean training dataset if the already trained model is not to be used.
    if not use_saved_models['clean']:
        attack.train(attack.converted_dataset)
        logging.info("=== Initial Model Training Completed ===")

    logging.info("=== Testing the model on the clean testing dataset ===")

    # Step 5: Test the model on the clean testing dataset
    converted_ba = attack.test(attack.converted_dataset[1])

    logging.info("=== Testing Completed ===")

    # Get current Unix timestamp
    unix_timestamp = int(time.time())


    # # If we are not using the already trained model, then save the model in the path.
    # if not use_saved_models['clean']:
    #     attack.model.save_model(clean_model_address)
    

    # Step 6: Select non-target samples from the training dataset
    D_non_target = attack.select_non_target_samples()

    # Step 7: Confidence-based sample ranking to create D_picked
    D_picked = attack.confidence_based_sample_ranking()


    # Step 8: Define the backdoor trigger
    attack.define_trigger()

    # Step 9: optimize the trigger
    attack.compute_mode() # compute the mode vector (if needed)


    if use_existing_trigger:
        attack.load_trigger() # load the already optimized trigger (if needed)
    else:
        attack.optimize_trigger() # optimize the trigger


    print("the trigger is: \n", attack.delta)

    print("the mode vector is: \n", attack.mode_vector)


    # Step 10: Construct the poisoned dataset
    
    poisoned_trainset, poisoned_train_samples = attack.construct_poisoned_dataset(attack.converted_dataset[0], epsilon=epsilon)
    poisoned_testset, poisoned_test_samples = attack.construct_poisoned_dataset(attack.converted_dataset[1], epsilon=1)
    attack.poisoned_dataset = (poisoned_trainset, poisoned_testset)
    attack.poisoned_samples = (poisoned_train_samples, poisoned_test_samples)


    # Save the poisoned dataset
    attack.save_poisoned_dataset()
    # load the poisoned dataset (uncomment if needed)
    # attack.load_poisoned_dataset()

    # If using xgboost, then we need to define a black box model.
    # Define the black box model
    if model_name == "xgboost":
        objective = "multi" if attack.data_obj.num_classes > 2 else "binary"
        black_box_model = model_dict[model_name](objective=objective, args=args)
        black_box_model.to(device)
        attack.model = black_box_model

    # for SAINT, we need the define a new model with is_numerical=False so 
    # the model can handle the categorical features as well.
    if attack.model.model_name == "SAINT":
        attack.model = SAINTModel(data_obj=attack.data_obj, is_numerical=False, args=args)


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

    logging.info("=== Training the model on the poisoned training dataset ===")
    # Step 12: Train the model on the poisoned training dataset
    converted = False if FTT and data_obj.cat_cols else True
    attack.train(reverted_poisoned_dataset, converted=converted)
    logging.info("=== Poisoned Training Completed ===")

    logging.info("=== Testing the model on the poisoned testing dataset and clean testing dataset ===")
    # Step 13: Test the model on the poisoned testing dataset and clean testing dataset
    asr = attack.test(reverted_poisoned_testset, converted=converted)
    cda = attack.test(clean_testset, converted=converted)
    logging.info("=== Testing Completed ===")

    # Step 14: Save the results to the csv file
    with open(csv_file_address, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([exp_num, dataset_name, model_name, target_label, epsilon, mu, beta, lambd, converted_ba, cda, asr])


    # creating and checking the path for saving and loading the poisoned models.
    poisoned_model_path = models_path / Path(f"poisoned")
    if not poisoned_model_path.exists():
        poisoned_model_path.mkdir(parents=True, exist_ok=True)
    poisoned_model_address = poisoned_model_path / Path(f"{model_name}_{data_obj.dataset_name}_{target_label}_{mu}_{beta}_{lambd}_{epsilon}_poisoned_model.pth")

    # Save the poisoned model with Unix timestamp in the filename
    if model_name == "ftt" and data_obj.cat_cols:
        attack.model.save_model(poisoned_model_address, model_type="original")
    else:
        attack.model.save_model(poisoned_model_address)






# ============================================


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
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for data loading. If not provided, uses number of CPU cores available.")

    # parse the arguments
    args = parser.parse_args()

    available_datasets = ["aci", "bm", "higgs", "credit_card", "covtype", "poker"]
    available_models = ["ftt", "tabnet", "saint", "xgboost"]

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
    
    # Determine number of workers for data loading
    if args.num_workers is None:
        args.num_workers = min(4, os.cpu_count())
        logging.info(f"No num_workers specified, using {args.num_workers} workers based on available CPU cores.")
    else:
        args.num_workers = max(0, min(args.num_workers, os.cpu_count()))
        logging.info(f"Using {args.num_workers} workers for data loading.")
        
    
    use_saved_models = {'clean': False, 'poisoned': False}
    use_existing_trigger = False

    # log all the arguments before running the experiment
    print("-"*100)
    logging.info(f"Running experiment with the following arguments: {args}")
    print("-"*100)
    print("\n")



    attack_step_by_step(args, use_saved_models, use_existing_trigger)


