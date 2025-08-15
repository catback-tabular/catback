# attack.py

from pathlib import Path
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader

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
# Attack Class Definition
# ============================================

class Attack:
    """
    The Attack class encapsulates the steps required to perform a backdoor attack on a neural network model.
    
    Attributes:
        model: The neural network model to be attacked.
        data_obj: The dataset object containing the data.
        converted_dataset (tuple): A tuple containing the training and testing datasets.
        device (torch.device): The device (CPU or GPU) on which computations are performed.
        target_label (int): The label of the target class for the backdoor attack.
        D_non_target (TensorDataset): The subset of the training dataset excluding samples with the target label.
        D_picked (TensorDataset): The subset of non-target samples selected based on confidence scores.
        mu (float): The fraction of non-target samples to pick based on confidence scores.
        delta (torch.Tensor): The universal trigger pattern to be added to inputs.
        min_X (torch.Tensor): The minimum values of each feature in the training dataset.
        max_X (torch.Tensor): The maximum values of each feature in the training dataset.
        mode_vector (torch.Tensor): The mode vector of the entire dataset \( D \).
        beta (float): Hyperparameter controlling the \( L_1 \) regularization term.
        lambd (float): Hyperparameter controlling the \( L_2 \) regularization term.
        poisoned_dataset (tuple): A tuple containing the poisoned training and testing datasets.
        poisoned_samples (tuple): A tuple containing the poisoned training and testing samples.

    """
    
    def __init__(self, device, model, data_obj, target_label, mu, beta, lambd, epsilon, args=None):
        """
        Initializes the class for performing a backdoor attack on a model.

        Parameters:
        device (torch.device): The device (CPU or GPU) on which computations are performed.
        model (TabNetModel): The TabNet model instance.
        data_obj (CovType): The CovType dataset instance.
        target_label (int): The label of the target class for the backdoor attack.
        mu (float, optional): Fraction of non-target samples to select based on confidence scores. Defaults to 0.2.
        beta (float, optional): Hyperparameter for \( L_1 \) regularization. Defaults to 0.1.
        lambd (float, optional): Hyperparameter for \( L_2 \) regularization. Defaults to 0.1.
        """

        # Initialize the dataset
        self.data_obj = data_obj
        self.converted_dataset = self.data_obj.get_normal_datasets() if not self.data_obj.cat_cols else self.data_obj.get_converted_dataset() 

        # Device
        self.device = device

        # Initialize the model
        self.model = model

        self.target_label = target_label

        # Initialize the D_non_target attribute to store non-target samples
        self.D_non_target = None

        # Initialize the D_picked attribute to store picked samples based on confidence
        self.D_picked = None

        # Set the mu parameter
        self.mu = mu


        # Initialize trigger pattern (delta) as None; to be defined in define_trigger()
        self.delta = None
        
        # Initialize min and max per feature; to be computed in compute_min_max()
        self.min_X = None
        self.max_X = None

        # Initialize the mode vector; to be computed in compute_mode()
        self.mode_vector = None
        
        # Set regularization hyperparameters
        self.beta = beta
        self.lambd = lambd
        self.num_workers = args.num_workers if args else 0
        self.batch_size = args.train_batch_size if args else 64
        self.epsilon = epsilon

         # Initialize the poisoned dataset and samples as None
        self.poisoned_dataset = (None, None)
        self.poisoned_samples = (None, None)

        
        logging.info(f"Attack initialized with target label: {self.target_label}")
        
    
    def train(self, dataset, converted=True):
        """
        Trains the model on the training set.

        Returns:
        None
        """
        train_dataset, val_dataset = dataset

        if self.model.model_name == "TabNet" or self.model.model_name == "XGBoost" or self.model.model_name == "CatBoost":

            # Convert training data to the required format
            X_train, y_train = self.data_obj._get_dataset_data(train_dataset)


            # Train the model
            self.model.fit(
                X_train=X_train, 
                y_train=y_train
            )
        elif self.model.model_name == "SAINT":
            X_train, y_train = self.data_obj._get_dataset_data(train_dataset)

            X_val, y_val = self.data_obj._get_dataset_data(val_dataset)
            
            self.model.fit(
                X_train=X_train, 
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
        elif self.model.model_name == "FTTransformer":
            if converted:
                self.model.fit_converted(train_dataset, val_dataset)
            else:
                self.model.fit(train_dataset, val_dataset)
        else:
            raise ValueError(f"Model {self.model.model_name} not supported.")
    
    def test(self, testset, converted=True):
        """
        Tests the trained model on the test set.

        Returns:
        accuracy (float): Accuracy of the model on the test set.
        """
        if self.model.model_name == "TabNet" or self.model.model_name == "XGBoost" or self.model.model_name == "CatBoost":
            # Convert test data to the required format
            X_test, y_test = self.data_obj._get_dataset_data(testset)

            # Make predictions on the test data
            preds = self.model.predict(X_test)

            # Calculate accuracy using NumPy
            accuracy = (preds == y_test).mean()
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            test_accuracy = accuracy * 100

        elif self.model.model_name == "FTTransformer":
            if converted:
                accuracy = self.model.predict_converted(testset)
            else:
                accuracy = self.model.predict(testset)
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            test_accuracy = accuracy * 100

        elif self.model.model_name == "SAINT":
            X_test, y_test = self.data_obj._get_dataset_data(testset)
            accuracy = self.model.predict(X_test, y_test)
            print(f"Test Accuracy: {accuracy}")
            test_accuracy = accuracy

        else:
            raise ValueError(f"Model {self.model.model_name} not supported.")

        return test_accuracy
    

    def select_non_target_samples(self) -> TensorDataset:
        """
        Selects non-target samples from the training dataset by excluding samples with the target label.
        
        Constructs the subset \( D_{\text{non-target}} \) which contains all samples not labeled as the target.
        
        Returns:
            D_non_target (TensorDataset): The subset of the training dataset excluding target samples.
        """
        logging.info("Selecting non-target samples from the training dataset...")
        
        # Extract training data
        X_train, y_train = self.data_obj._get_dataset_data(self.converted_dataset[0])
        
        # Convert to NumPy arrays for processing
        X_train_np = X_train
        y_train_np = y_train
        
        # Identify indices where the label is not equal to the target label
        non_target_indices = np.where(y_train_np != self.target_label)[0]
        
        # Select non-target samples based on identified indices
        X_non_target = X_train_np[non_target_indices]
        y_non_target = y_train_np[non_target_indices]
        
        logging.info(f"Selected {len(X_non_target)} non-target samples out of {len(X_train_np)} total training samples.")
        
        # Convert selected non-target samples to PyTorch tensors
        X_non_target_tensor = torch.tensor(X_non_target, dtype=torch.float32)
        y_non_target_tensor = torch.tensor(y_non_target, dtype=torch.long)
        
        # Create a TensorDataset for the non-target samples
        self.D_non_target = TensorDataset(X_non_target_tensor, y_non_target_tensor)
    
        return self.D_non_target
    

    def confidence_based_sample_ranking(self, batch_size=None) -> TensorDataset:
        """
        Performs confidence-based sample ranking to select the top mu fraction of non-target samples
        based on the model's confidence in predicting the target class.

        Steps:
            1. Evaluate the model on D_non_target to obtain softmax confidence scores for the target class.
            2. Pair each input with its confidence score to form D_conf.
            3. Sort D_conf in descending order based on confidence scores.
            4. Select the top mu fraction of samples to create D_picked.

        Args:
            batch_size (int, optional): Number of samples per batch for evaluation. Defaults to 64.
        
        Returns:
            D_picked (TensorDataset): The subset of non-target samples selected based on confidence scores.
        """
        if self.D_non_target is None:
            raise ValueError("D_non_target is not initialized. Please run select_non_target_samples() first.")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        logging.info("Starting confidence-based sample ranking...")
        
        # Create a DataLoader for D_non_target
        pin_memory = torch.cuda.is_available()
        non_target_loader = DataLoader(self.D_non_target, batch_size=batch_size, shuffle=False,
                                     num_workers=self.num_workers, pin_memory=pin_memory,
                                     persistent_workers=self.num_workers > 0)
        
        # Initialize lists to store confidence scores and corresponding indices
        confidence_scores = []
        sample_indices = []
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(non_target_loader):
                # Move data to the appropriate device
                X_batch = X_batch.to(self.device)
                
                # Forward pass through the model to get logits
                logits = self.model.forward(X_batch)
                
                # Apply softmax to obtain probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Extract the probability of the target class for each sample in the batch
                # Assuming target_label is zero-indexed
                s_batch = probabilities[:, self.target_label]
                
                # Move probabilities to CPU and convert to NumPy
                s_batch_np = s_batch.cpu().numpy()
                
                # Calculate the absolute sample indices
                # Assuming DataLoader iterates in order without shuffling
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(s_batch_np)
                batch_indices = np.arange(start_idx, end_idx)
                
                # Append to the lists
                confidence_scores.extend(s_batch_np)
                sample_indices.extend(batch_indices)
        
        # Convert lists to NumPy arrays for efficient processing
        confidence_scores = np.array(confidence_scores)
        sample_indices = np.array(sample_indices)
        
        # Pair each sample index with its confidence score
        D_conf = list(zip(sample_indices, confidence_scores))
        
        # Sort D_conf in descending order based on confidence scores
        D_conf_sorted = sorted(D_conf, key=lambda x: x[1], reverse=True)
        
        # Determine the number of samples to pick based on mu
        total_non_target = len(D_conf_sorted)
        top_mu_count = int(self.mu * total_non_target)
        logging.info(f"Selecting top {self.mu*100:.1f}% ({top_mu_count}) samples based on confidence scores.")
        
        # Select the top mu fraction of samples
        D_picked_indices = [idx for idx, score in D_conf_sorted[:top_mu_count]]
        
        # Retrieve the corresponding samples from D_non_target
        X_picked = []
        y_picked = []
        for idx in D_picked_indices:
            X_picked.append(self.D_non_target[idx][0].cpu())
            y_picked.append(self.D_non_target[idx][1].cpu())
        
        # Convert lists to tensors
        X_picked_tensor = torch.stack(X_picked)
        y_picked_tensor = torch.tensor(y_picked, dtype=torch.long)
        
        # Create a TensorDataset for the picked samples
        self.D_picked = TensorDataset(X_picked_tensor, y_picked_tensor)
        
        logging.info(f"Selected {len(self.D_picked)} samples for further processing.")
        
        return self.D_picked
    

    def compute_min_max(self):
        """
        Computes the minimum and maximum values for each feature in the training dataset.
        
        These values are used to ensure that the trigger pattern \( \delta \) does not push any feature
        beyond its valid range when applied.
        """
        logging.info("Computing min and max values for each feature in the training dataset...")
        
        # Extract training data
        X_train, _ = self.data_obj._get_dataset_data(self.converted_dataset[0])
        
        # Convert to NumPy for efficient computation
        X_train_np = X_train
        
        # Compute min and max per feature
        min_vals = X_train_np.min(axis=0)
        max_vals = X_train_np.max(axis=0)
        
        # Convert to PyTorch tensors and move to device
        self.min_X = torch.tensor(min_vals, dtype=torch.float32).to(self.device)
        self.max_X = torch.tensor(max_vals, dtype=torch.float32).to(self.device)
        
        logging.info("Min and max values computed successfully.")
    
    def define_trigger(self):
        """
        Defines the universal trigger pattern \( \delta \) to be added to inputs.
        
        Initializes \( \delta \) as a PyTorch tensor with requires_grad=True, allowing it to be optimized later.
        The trigger is initialized with zeros, but it can be initialized with small random values if desired.
        """
        logging.info("Defining the universal trigger pattern (delta)...")
        
        if self.min_X is None or self.max_X is None:
            self.compute_min_max()
        
        # Determine the number of features
        num_features = self.min_X.shape[0]
        
        # Initialize delta with zeros; alternatively, use small random values
        self.delta = torch.zeros(num_features, device=self.device, requires_grad=True)
        
        # Alternatively, initialize with small random values (uncomment below if desired)
        # self.delta = torch.randn(num_features, device=self.device) * 0.01
        # self.delta.requires_grad = True
        
        logging.info(f"Trigger pattern (delta) initialized with shape: {self.delta.shape}")
    
    def apply_trigger(self, X):
        """
        Applies the universal trigger pattern \( \delta \) to a batch of inputs, ensuring that
        the modified inputs remain within valid feature ranges via clipping.
        
        Args:
            X (torch.Tensor): A batch of input samples with shape (batch_size, d).
        
        Returns:
            X_hat (torch.Tensor): The batch of backdoored inputs after applying the trigger and clipping.
        """
        if self.delta is None:
            raise ValueError("Trigger pattern (delta) is not defined. Please run define_trigger() first.")
        
        # Add delta to the input batch
        X_hat = X + self.delta.unsqueeze(0)  # Expand delta to match batch size
        
        # Apply clipping to ensure each feature remains within its valid range
        X_hat = torch.clamp(X_hat, min=self.min_X, max=self.max_X)
        
        return X_hat
    

    def compute_mode(self):

        """
        Computes the mode vector of the entire training dataset \( D \) known by the attacker.
        
        The mode vector \( \text{Mode}(X) \) has each element \( \text{Mode}(X)^{(j)} \) as the mode of feature \( j \).
        
        Returns:
            mode_vector (torch.Tensor): A tensor containing the mode value for each feature.
        """

        logging.info("Computing mode vector of the entire training dataset...")
        
        # Extract training data
        X_train, _ = self.data_obj._get_dataset_data(self.converted_dataset[0])
        
        # Convert to Pandas DataFrame for mode computation
        X_train_df = pd.DataFrame(X_train)
        
        # Compute mode for each feature
        mode_values = X_train_df.mode().iloc[0].values  # mode() returns a DataFrame; take the first row
        
        # Convert to PyTorch tensor and move to device
        self.mode_vector = torch.tensor(mode_values, dtype=torch.float32).to(self.device)
        
        logging.info("Mode vector computed successfully.")
        
        return self.mode_vector
    
 


    def optimize_trigger(self, num_epochs=200, learning_rate=0.0001, batch_size=None, verbose=True, patience=30):
        """
        Optimizes the universal trigger pattern \( \delta \) by minimizing the loss function over D_picked.

        The loss function is defined as:
        \[
        \mathcal{L}(\delta) = \frac{1}{|D_{\text{picked}}|} \sum_{(x_i, y_i) \in D_{\text{picked}}} \left[ -\log f_t( \hat{x}_i ) + \beta \| \hat{x}_i - \text{Mode}(X) \|_1 + \lambda \| \hat{x}_i - \text{Mode}(X) \|_2^2 \right]
        \]

        This optimization balances three objectives:
            1. Maximizing the model's confidence in predicting the target class \( t \) for the backdoored inputs.
            2. Ensuring the trigger pattern \( \delta \) keeps the modified inputs close to common data patterns (via the mode) to enhance stealthiness.
            3. Regularizing \( \delta \) to prevent large perturbations that could be easily detected.

        Args:
            num_epochs (int, optional): Number of optimization epochs. Defaults to 200.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
            batch_size (int, optional): Number of samples per batch for optimization. Defaults to 64.
            verbose (bool, optional): If True, logs loss every 10 epochs. Defaults to True.
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 30.
        
        Returns:
            None
        """
        if self.D_picked is None:
            raise ValueError("D_picked is not initialized. Please run confidence_based_sample_ranking() first.")
        
        if self.delta is None:
            raise ValueError("Trigger pattern (delta) is not defined. Please run define_trigger() first.")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        logging.info("Starting optimization of the trigger pattern (delta)...")
        logging.info(f"Using batch size: {batch_size} for trigger optimization")

        
        # Define the optimizer for delta
        optimizer = torch.optim.Adam([self.delta], lr=learning_rate)
        
        # Define the DataLoader for D_picked
        pin_memory = torch.cuda.is_available()
        picked_loader = DataLoader(self.D_picked, batch_size=batch_size, shuffle=True,
                                 num_workers=self.num_workers, pin_memory=pin_memory,
                                 persistent_workers=self.num_workers > 0)
        
        # Ensure that the mode vector is computed
        if self.mode_vector is None:
            self.compute_mode()
        
        # Set the model to evaluation mode to disable dropout, etc.
        self.model.eval()
        
        # Early stopping variables
        best_loss = float('inf')
        best_delta = None
        counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in picked_loader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Apply the trigger to the inputs
                X_hat = self.apply_trigger(X_batch)
                
                # Forward pass to get logits
                logits = self.model.forward(X_hat)
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=1)
                
                # Extract the probability for the target class
                f_t = probabilities[:, self.target_label]
                
                # Compute the negative log-likelihood loss
                nll_loss = -torch.log(f_t + 1e-8).mean()
                
                # Compute the L1 and L2 regularization terms
                l1_loss = torch.norm(X_hat - self.mode_vector, p=1)
                l2_loss = torch.norm(X_hat - self.mode_vector, p=2)**2
                
                # Total loss
                loss = nll_loss + self.beta * l1_loss + self.lambd * l2_loss
                
                # Backward pass
                loss.backward()
                
                # Update delta
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()
            
            # Calculate average loss for this epoch
            avg_loss = epoch_loss / len(picked_loader)
            
            # Logging
            if verbose and epoch % 10 == 0:
                logging.info(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
            
            # Check if this is the best loss so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Store a copy of the current delta
                best_delta = self.delta.clone().detach()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch} with best loss: {best_loss:.4f}")
                    # Restore the best delta
                    self.delta = best_delta
                    break
        
        # If we didn't trigger early stopping, make sure we use the best delta
        if counter < patience and best_delta is not None:
            self.delta = best_delta
            
        logging.info(f"Trigger pattern (delta) optimization completed with final loss: {best_loss:.4f}")
        self.save_trigger()







    def save_trigger(self, filepath=None):
        """
        Saves the optimized trigger pattern \( \delta \) to disk for future use.

        Args:
            filepath (str, optional): Path to save the trigger file. Defaults to 'delta.pt'.
        """
        if self.delta is None:
            raise ValueError("Trigger pattern (delta) is not defined. Please run define_trigger() first.")
        
        # Detach delta from the computation graph and move to CPU
        delta_cpu = self.delta.detach().cpu()

        if filepath is None:
            file_dir = Path("./saved_triggers")
            # Create the directory if it doesn't exist
            file_dir.mkdir(parents=True, exist_ok=True)
            # Define the default filepath
            filepath = file_dir / f"{self.data_obj.dataset_name}_{self.model.model_name}_{self.target_label}_{self.mu}_{self.beta}_{self.lambd}_delta.pt"
        
        # Save the delta tensor
        torch.save(delta_cpu, filepath)
        
        logging.info(f"Optimized trigger pattern (delta) saved to '{filepath}'.")

    def load_trigger(self, filepath=None):
        """
        Loads a previously saved trigger pattern \( \delta \) from disk.

        Args:
            filepath (str, optional): Path to the trigger file. Defaults to 'delta.pt'.
        
        Raises:
            FileNotFoundError: If the specified trigger file does not exist.
        """
        if filepath is None:
            file_dir = Path("./saved_triggers")
            # Define the default filepath
            filepath = file_dir / f"{self.data_obj.dataset_name}_{self.model.model_name}_{self.target_label}_{self.mu}_{self.beta}_{self.lambd}_delta.pt"

        if not filepath.exists():
            raise FileNotFoundError(f"The trigger file '{filepath}' does not exist.")
        
        # Load the delta tensor and move to device
        self.delta = torch.load(filepath, map_location=self.device)
        
        # Ensure that delta requires gradient for further optimization if needed
        self.delta.requires_grad = True
        
        logging.info(f"Trigger pattern (delta) loaded from '{filepath}'.")



    def construct_poisoned_dataset(self, dataset, epsilon=0.1, random_state=None):
        """
        Constructs the poisoned dataset by injecting the optimized trigger into a fraction epsilon of the dataset D.
        
        Steps:
            1. Determine the number of samples to poison based on epsilon.
            2. Randomly select epsilon fraction of the training dataset to poison.
            3. Apply the optimized trigger delta* to the selected samples.
            4. Relabel the poisoned samples to the target class t.
            5. Combine the poisoned samples with the remaining clean samples to form the final poisoned dataset D'.
            6. Optionally, revert the numerical encoding to original categorical features for analysis or storage.

        Args:
            dataset (TensorDataset): The dataset to poison.
            epsilon (float, optional): Fraction of the dataset to poison. Must be in (0, 1]. Defaults to 0.1.
            random_state (int, optional): Random state for reproducibility. Defaults to None.
        
        Returns:
            poisoned_dataset (TensorDataset): The final poisoned training dataset \( D' \).
            poisoned_samples (TensorDataset): The set of poisoned samples.
        """


        if self.D_picked is None:
            raise ValueError("D_picked is not initialized. Please run confidence_based_sample_ranking() first.")
        
        if self.delta is None:
            raise ValueError("Trigger pattern (delta) is not defined. Please run define_trigger() first.")
        
        if not (0 < epsilon <= 1):
            raise ValueError("Epsilon must be in the range (0, 1].")
        
        logging.info(f"Starting construction of the poisoned dataset with epsilon={epsilon}...")
        
        # Extract the original training dataset
        X_train, y_train = self.data_obj._get_dataset_data(dataset)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        # Determine the number of samples to poison
        N = len(X_train_tensor)
        N_poison = int(epsilon * N)
        logging.info(f"Number of samples to poison: {N_poison} out of {N} total training samples.")
        
        # Set random seed for reproducibility if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Randomly select indices to poison
        poison_indices = np.random.choice(N, N_poison, replace=False)
        logging.info(f"Selected {len(poison_indices)} random samples for poisoning.")
        
        # Create a mask for all indices
        mask = np.ones(N, dtype=bool)
        mask[poison_indices] = False
        
        # Separate clean and selected samples
        X_clean = X_train_tensor[mask]
        y_clean = y_train_tensor[mask]
        
        X_selected = X_train_tensor[poison_indices]
        y_selected = y_train_tensor[poison_indices]  # Original labels, which will be overwritten
        
        # Apply the optimized trigger to the selected samples
        X_poisoned = self.apply_trigger(X_selected)

        # Round the r_jl values of the poisoned samples
        if self.data_obj.cat_cols:
            X_poisoned = self.data_obj.round_rjl(X_poisoned)
        else:
            X_poisoned = X_poisoned.clone().detach().cpu()
        
        # Relabel the poisoned samples to the target class
        y_poisoned = torch.full((N_poison,), self.target_label, dtype=torch.long).to(self.device)
        
        logging.info("Applied trigger to selected samples and relabeled them to the target class.")
        
        # Combine clean and poisoned samples to form the final poisoned dataset D'
        # Ensure all tensors are on the same device before concatenation
        device = self.device  # Assuming self.device is defined
        X_clean = X_clean.to(device)
        X_poisoned = X_poisoned.to(device)
        y_clean = y_clean.to(device)
        y_poisoned = y_poisoned.to(device)

        # Concatenate tensors on the same device
        X_poisoned_dataset = torch.cat((X_clean, X_poisoned), dim=0)
        y_poisoned_dataset = torch.cat((y_clean, y_poisoned), dim=0)
        
        # Shuffle the poisoned dataset to mix poisoned and clean samples
        indices = torch.randperm(X_poisoned_dataset.size(0))
        X_poisoned_dataset = X_poisoned_dataset[indices]
        y_poisoned_dataset = y_poisoned_dataset[indices]
        
        
        # Create the poisoned TensorDataset
        poisoned_dataset = TensorDataset(X_poisoned_dataset, y_poisoned_dataset)
        poisoned_samples = TensorDataset(X_poisoned, y_poisoned)
        
        logging.info(f"Poisoned dataset constructed with {len(poisoned_dataset)} samples.")


        #####################################
        # Select a subset of poisoned samples to display
        # num_samples_to_show = min(5, N_poison)  # Show up to 5 samples or the total number of poisoned samples if less
        # selected_indices = torch.randperm(N_poison)[:num_samples_to_show]
        
        # # Extract the selected samples before and after poisoning
        # X_selected_before_poisoning = X_selected[selected_indices]
        # X_selected_after_poisoning = X_poisoned[selected_indices]
        
        # # Calculate the delta (difference) between the original and poisoned samples
        # delta = self.delta
        
        # # Print the selected samples before and after poisoning, and the delta
        # for i in range(num_samples_to_show):
        #     print(f"Sample {i + 1} before poisoning:\n{X_selected_before_poisoning[i]}")
        #     print(f"Sample {i + 1} after poisoning:\n{X_selected_after_poisoning[i]}")
        #     print("-" * 50)
        # print(delta)
        
        # logging.info("Displayed selected samples before and after poisoning, along with their deltas.")
        #####################################

        
        return poisoned_dataset, poisoned_samples


    def save_poisoned_dataset(self, filepath=None):
        """
        Saves the poisoned dataset \( D' \) to disk for future use.

        Args:
            filepath (str, optional): Path to save the poisoned dataset file. Defaults to 'poisoned_dataset.pt'.
        """
        if self.poisoned_dataset[0] is None or self.poisoned_dataset[1] is None:
            raise ValueError("Poisoned dataset is not constructed. Please run construct_poisoned_dataset() first and save train and test datasets in a tuple.")
        if self.poisoned_samples[0] is None or self.poisoned_samples[1] is None:
            raise ValueError("Poisoned samples are not constructed. Please run construct_poisoned_dataset() first and save train and test samples in a tuple.")
        
        if filepath is None:
            file_dir = Path("./saved_datasets")
            # Create the directory if it doesn't exist
            file_dir.mkdir(parents=True, exist_ok=True)
            # Define the default filepath
            filepath = file_dir / f"{self.data_obj.dataset_name}_{self.model.model_name}_{self.target_label}_{self.mu}_{self.beta}_{self.lambd}_{self.epsilon}_poisoned_dataset.pt"
        
        # Save the poisoned dataset tensors
        torch.save({
            'poisoned_trainset': self.poisoned_dataset[0],
            'poisoned_testset': self.poisoned_dataset[1],
            'poisoned_train_samples': self.poisoned_samples[0],
            'poisoned_test_samples': self.poisoned_samples[1]
        }, filepath)
        
        logging.info(f"Poisoned dataset saved to '{filepath}'.")

    def load_poisoned_dataset(self, filepath=None):
        """
        Loads a previously saved poisoned dataset \( D' \) from disk.

        Args:
            filepath (str, optional): Path to the poisoned dataset file. Defaults to 'poisoned_dataset.pt'.
        
        Raises:
            FileNotFoundError: If the specified poisoned dataset file does not exist.
        """
        if filepath is None:
            file_dir = Path("./saved_datasets")
            # Define the default filepath
            filepath = file_dir / f"{self.data_obj.dataset_name}_{self.model.model_name}_{self.target_label}_{self.mu}_{self.beta}_{self.lambd}_{self.epsilon}_poisoned_dataset.pt"

        if not filepath.exists():
            raise FileNotFoundError(f"The poisoned dataset file '{filepath}' does not exist.")
        
        # Load the poisoned dataset tensors
        data = torch.load(filepath, map_location=self.device)
        
        
        # Create the TensorDataset
        self.poisoned_dataset = (data['poisoned_trainset'], data['poisoned_testset'])
        self.poisoned_samples = (data['poisoned_train_samples'], data['poisoned_test_samples'])
        
        logging.info(f"Poisoned dataset loaded from '{filepath}' with {len(self.poisoned_dataset[0])} samples in trainset and {len(self.poisoned_dataset[1])} samples in testset.")
