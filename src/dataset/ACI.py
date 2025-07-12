from pathlib import Path
import pickle
import random
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import shap



class ACI:
    """
    This class is used to load the ACI dataset and convert it into a format suitable for training a model.
    """

    def __init__(self, test_size=0.2, random_state=None, batch_size=64):

        self.dataset_name = "aci"
        self.num_classes = 2

        # Define the categorical and numerical columns
        self.cat_cols = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
        
        self.num_cols = ['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']

        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

        # Load the ACI dataset from shap
        X, y = shap.datasets.adult()

        # Retrieve the feature names from the dataset: in the same order as they appear in the dataset
        self.feature_names = X.columns.tolist()

        self.column_idx = {col: idx for idx, col in enumerate(self.feature_names)}
        self.cat_cols_idx = [self.column_idx[col] for col in self.cat_cols]
        self.num_cols_idx = [self.column_idx[col] for col in self.num_cols]

        # Store original dataset for reference
        self.X_original = X.copy()
        # Convert boolean target to binary (0 and 1)
        self.y = y.astype(int).copy()


        # Convert categorical columns using OrdinalEncoder
        # This transforms categorical string labels into integer encodings
        ordinal_encoder = OrdinalEncoder()
        self.X_encoded = X.copy()
        self.X_encoded.loc[:, self.cat_cols] = ordinal_encoder.fit_transform(X[self.cat_cols])


        # For training the FTT model, I need to know the number of unique categories in each categorical feature as a tuple
        self.FTT_n_categories = tuple(len(self.X_encoded[col].unique()) for col in self.cat_cols)
        
        # Apply StandardScaler to numerical features to standardize them
        self.std_scaler = StandardScaler()
        self.X_encoded[self.num_cols] = self.std_scaler.fit_transform(self.X_encoded[self.num_cols])

        # Initialize a dictionary to store the primary mappings for each categorical feature
        self.primary_mappings = {col: {} for col in self.cat_cols}

        # Initialize a dictionary to store the adaptive Delta r for each categorical feature
        self.delta_r_values = {col: 0.0 for col in self.cat_cols}

        # Initialize a dictionary to store the hierarchical mappings for each categorical feature
        self.hierarchical_mappings = {col: {} for col in self.cat_cols}
        
        # Initialize a dictionary to store the lookup tables for reverse mapping
        self.lookup_tables = {col: {} for col in self.cat_cols}


    def get_normal_datasets(self, dataloader=False, batch_size=None, test_size=None, random_state=None):

        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size

        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(self.X_encoded, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        
        # Further split the temporary set into validation and test sets
        # val_size_adjusted = self.val_size / (1 - test_size)  # Adjust validation size based on remaining data
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
        
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create TensorDatasets for each split
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the Datasets for training and test sets
        if dataloader:
            return train_loader, test_loader
        else:
            return train_dataset, test_dataset



    
    def get_normal_datasets_FTT(self, dataloader=False, batch_size=None, test_size=None, random_state=None):
        """
        Returns the datasets for training and testing the FTT model. This method is the same as previous method: get_normal_datasets, but with one difference:
        before feeding the X_encoded to the train_test_split, we seperate the categorical and numerical features into two different frames called X_encoded_cat and X_encoded_num. then 
        we feed all 3 of them to the train_test_split and the rest of the process is the same as the previous method.
        """

        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size

        X_encoded_cat = self.X_encoded[self.cat_cols]
        X_encoded_num = self.X_encoded[self.num_cols]

        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(X_encoded_cat, X_encoded_num, self.y, test_size=test_size, random_state=random_state, stratify=self.y)


        # convert the data to PyTorch tensors
        X_train_cat_tensor = torch.tensor(X_train_cat.values, dtype=torch.long)
        X_test_cat_tensor = torch.tensor(X_test_cat.values, dtype=torch.long)
        X_train_num_tensor = torch.tensor(X_train_num.values, dtype=torch.float32)
        X_test_num_tensor = torch.tensor(X_test_num.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create TensorDatasets for each split
        train_dataset = TensorDataset(X_train_cat_tensor, X_train_num_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_cat_tensor, X_test_num_tensor, y_test_tensor)

        # Create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Return the Datasets for training and test sets
        if dataloader:
            return train_loader, test_loader
        else:
            return train_dataset, test_dataset



    def get_normal_datasets_ohe(self, dataloader=False, batch_size=None, test_size=None, random_state=None):
        """
        Returns the datasets for training and testing. This method is the same as the previous method: get_normal_datasets, but with one difference:
        before feeding the X_encoded to the train_test_split, we convert the categorical features to one-hot encoded features using OneHotEncoder.
        """

        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size

        X_encoded_copy = self.X_encoded.copy()

        # Convert the categorical features to one-hot encoded features using OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse_output=False)  # Ensure the output is a dense array
        X_encoded_ohe = onehot_encoder.fit_transform(X_encoded_copy[self.cat_cols])

        # Create a DataFrame from the one-hot encoded data
        ohe_columns = onehot_encoder.get_feature_names_out(self.cat_cols)
        X_encoded_ohe_df = pd.DataFrame(X_encoded_ohe, columns=ohe_columns, index=X_encoded_copy.index)

        # Concatenate the one-hot encoded columns with the numerical columns
        X_encoded_copy = pd.concat([X_encoded_ohe_df, X_encoded_copy[self.num_cols]], axis=1)

        # Now that X_encoded_copy is extended in its length, we need to update the feature_names
        self.feature_names = X_encoded_copy.columns.tolist()


        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded_copy, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        
        # Further split the temporary set into validation and test sets
        # val_size_adjusted = self.val_size / (1 - test_size)  # Adjust validation size based on remaining data
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
        
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create TensorDatasets for each split
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the Datasets for training and test sets
        if dataloader:
            return train_loader, test_loader
        else:
            return train_dataset, test_dataset



    def compute_primary_frequency_mapping(self):
        """
        Computes the frequency of each category in the categorical features and assigns
        a unique numerical representation r_jl based on the provided formula:
        
        r_jl = (c_max_j - c_jl) / (c_max_j - 1)
        
        where:
            - c_jl is the count of category v_jl in feature j
            - c_max_j is the maximum count among all categories in feature j
        """
        # Iterate over each categorical feature
        for col in self.cat_cols:
            # Get the column data
            col_data = self.X_encoded[col]
            
            # Calculate the frequency (count) of each unique category in the feature
            freq_counts = col_data.value_counts().sort_index()  # Sort by category index for consistency
            
            # Identify the maximum frequency in the current feature
            c_max_j = freq_counts.max()
            
            # Handle the edge case where c_max_j == 1 to avoid division by zero
            if c_max_j == 1:
                # If all categories have the same frequency, assign r_jl = 1 for all
                r_jl = {category: 1.0 for category in freq_counts.index}
                self.primary_mappings[col] = r_jl
                print(f"Primary Mapping for feature '{col}': All categories have r_jl = 1.0\n")
                continue  # Move to the next feature
            
            # Initialize a dictionary to store r_jl for each category in the feature
            r_jl = {}
            
            # Compute r_jl for each category using the provided formula
            for category, count in freq_counts.items():
                r_value = (c_max_j - count) / (c_max_j - 1)
                r_jl[category] = round(r_value, 5)  # Rounded to 5 decimal places for precision
            
            # Store the mapping in the primary_mappings dictionary
            self.primary_mappings[col] = r_jl
            
            # Optional: Print the mapping for verification
            print(f"Primary Mapping for feature '{col}':")
            for category, r_value in r_jl.items():
                print(f"  Category {int(category)}: r_jl = {r_value}")
            print("\n")  # Add a newline for better readability

    def get_primary_mappings(self):
        """
        Returns the primary frequency-based mappings for all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is another dictionary mapping category to r_jl.
        """
        return self.primary_mappings
    


    def compute_adaptive_delta_r(self):
        """
        Determines the adaptive Delta r for each categorical feature based on the smallest
        decimal precision in the primary mapping. This ensures that Delta r is one order
        of magnitude smaller than the smallest decimal precision in Delta r_min.
        """

        self.largest_p = 0
        # Iterate over each categorical feature
        for col in self.cat_cols:
            r_jl_mapping = self.primary_mappings[col]
            
            # If all r_jl are 1.0, skip Delta r computation as mapping cannot be refined
            if all(r == 1.0 for r in r_jl_mapping.values()):
                print(f"Feature '{col}' has all categories with r_jl = 1.0. Skipping Delta r computation.\n")
                continue  # Move to the next feature
            
            # Extract unique r_jl values and sort them in ascending order
            unique_r_jl = sorted(set(r_jl_mapping.values()))
            
            # Compute Delta r_min: the smallest difference between consecutive r_jl values
            delta_r_min = min(
                unique_r_jl[i+1] - unique_r_jl[i] for i in range(len(unique_r_jl) -1)
            )
            
            # Determine the first non-zero decimal place in Delta r_min
            # Convert Delta r_min to string to identify decimal places
            delta_r_min_str = f"{delta_r_min:.10f}"  # Format to 10 decimal places
            # Remove leading '0.' to focus on decimal digits
            decimal_part = delta_r_min_str.split('.')[1]
            
            # Initialize p to None
            p = None
            
            # Iterate through the decimal digits to find the first non-zero digit
            for idx, digit in enumerate(decimal_part, start=1):
                if digit != '0':
                    p = idx
                    break
            
            # If p is not found (delta_r_min is 0), set a default small p
            if p is None:
                p = 0  # Arbitrary large p to set Delta r very small
                print(f"Delta r_min for feature '{col}' is 0. Setting p={p} and Delta r accordingly.")
            
            # Set Delta r = 10^{-(p + 1)}
            delta_r = 10 ** (-(p + 1))

            # Update the largest p value
            if p > self.largest_p:
                self.largest_p = p
            
            # Store Delta r in the delta_r_values dictionary
            self.delta_r_values[col] = delta_r
            
            print(f"For feature '{col}':")
            print(f"  Delta r_min = {delta_r_min}")
            print(f"  Determined p = {p}")
            print(f"  Set Delta r = {delta_r}\n")
    
    def get_adaptive_delta_r(self):
        """
        Returns the adaptive Delta r values for all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is the corresponding Delta r.
        """
        return self.delta_r_values
        


    def compute_hierarchical_mapping(self):
        """
        Identifies tied categories within each categorical feature and assigns unique
        r'_jl values by adding incremental multiples of Delta r based on secondary ordering.
        
        Steps:
            1. Detect tied categories (categories with identical r_jl).
            2. Apply secondary ordering (e.g., alphabetical order) to tied categories.
            3. Assign unique r'_jl by adding (k-1)*Delta r to r_jl.
            4. Ensure that r'_jl <= 1. If not, adjust Delta r or redistribute offsets.
        """
        # Iterate over each categorical feature
        for col in self.cat_cols:
            primary_mapping = self.primary_mappings[col]
            delta_r = self.delta_r_values[col]
            
            # Skip features where all r_jl = 1.0
            if delta_r == 0.0:
                # Assign r'_jl = r_jl for all categories
                self.hierarchical_mappings[col] = primary_mapping.copy()
                # Build lookup table
                for category, r_value in primary_mapping.items():
                    self.lookup_tables[col][r_value] = category
                print(f"Hierarchical Mapping for feature '{col}': All categories have r'_jl = {primary_mapping[list(primary_mapping.keys())[0]]}\n")
                continue  # Move to the next feature
            
            # Invert the primary mapping to find categories with the same r_jl
            inverted_mapping = {}
            for category, r_value in primary_mapping.items():
                inverted_mapping.setdefault(r_value, []).append(category)
            
            # Initialize hierarchical mapping for the current feature
            hierarchical_mapping = {}
            
            # Iterate over each unique r_jl value
            for r_value, categories in inverted_mapping.items():
                if len(categories) == 1:
                    # No tie, assign r'_jl = r_jl
                    category = categories[0]
                    hierarchical_mapping[category] = round(float(r_value), self.largest_p + 1)
                else:
                    # Tie detected, need to resolve
                    tied_categories = categories.copy()
                    
                    # Apply secondary ordering: sort categories numerically (since categories are encoded as integers)
                    # If original categories are strings, sort alphabetically. Assuming encoded as integers here.
                    tied_categories_sorted = sorted(tied_categories)
                    
                    # Assign unique r'_jl by adding (k-1)*Delta r
                    for idx, category in enumerate(tied_categories_sorted, start=1):
                        r_prime = r_value + (idx - 1) * delta_r

                        # # Ensure r'_jl <=1
                        # if r_prime > 1.0:
                        #     # Adjust Delta r or redistribute offsets
                        #     # Option 1: Reduce Delta r dynamically (not implemented here for simplicity)
                        #     # Option 2: Redistribute offsets evenly within the available range
                            
                        #     # Calculate available range
                        #     available_range = 1.0 - r_value
                        #     # Number of tied categories
                        #     k = len(tied_categories_sorted)
                        #     # New Delta r to fit within available range
                        #     if k > 1:
                        #         adjusted_delta_r = available_range / k
                        #     else:
                        #         adjusted_delta_r = 0.0  # Only one category, no adjustment needed
                            
                        #     # Recompute r'_jl with adjusted Delta r
                        #     r_prime = r_value + (idx - 1) * adjusted_delta_r
                            
                        #     # Update Delta r for future assignments (optional)
                        #     # self.delta_r_values[col] = adjusted_delta_r
                            
                        #     print(f"Adjusted Delta r for feature '{col}' due to overflow:")
                        #     print(f"  Original Delta r = {delta_r}")
                        #     print(f"  Adjusted Delta r = {adjusted_delta_r}")
                            
                        # Round r'_jl to the largest p + 1 decimal places for consistency
                        r_prime = round(r_prime, self.largest_p + 1)
                        
                        # Assign r'_jl to the category
                        hierarchical_mapping[category] = r_prime
                        
            # Store the hierarchical mapping
            self.hierarchical_mappings[col] = hierarchical_mapping
            
            # Build the lookup table for reverse mapping
            for category, r_prime in hierarchical_mapping.items():
                # Ensure the category values and r_prime values are stored correctly
                self.lookup_tables[col][r_prime] = category

            
            
            # Optional: Print the hierarchical mapping for verification
            print(f"Hierarchical Mapping for feature '{col}':")
            for category, r_prime in hierarchical_mapping.items():
                print(f"  Category {int(category)}: r'_jl = {r_prime}")
            print("\n")  # Add a newline for better readability

        # Verify that the hierarchical mappings and lookup tables are consistent
        for col in self.hierarchical_mappings:
            hierarchical_mapping = self.hierarchical_mappings[col]
            lookup_table = self.lookup_tables[col]
            
            for category, r_prime in hierarchical_mapping.items():

                # Check if r_prime is not in the keys of the lookup table
                if r_prime not in lookup_table:
                    raise KeyError(f"r_prime value {r_prime} for category {category} in column '{col}' is not found in the lookup table.")
                # Check if the r_prime value in the lookup table matches the hierarchical mapping
                if lookup_table[r_prime] != category:
                    raise ValueError(f"Mismatch in column '{col}': category {category} has r_prime {r_prime} in hierarchical mapping, but lookup table has category {lookup_table[r_prime]} for the same r_prime.")
                

                
        
    def get_hierarchical_mappings(self):
        """
        Returns the hierarchical mappings for all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is another dictionary mapping category to r'_jl.
        """
        return self.hierarchical_mappings
    
    def get_lookup_tables(self):
        """
        Returns the lookup tables for reverse mapping of all categorical features.
        
        Returns:
            dict: A dictionary where each key is a categorical feature name and the value
                  is another dictionary mapping r'_jl to category.
        """
        return self.lookup_tables
    
    
    def Conv(self):
        """
        Applies the hierarchical mapping to convert categorical features in the dataset.
        
        Args:
            D_original (pd.DataFrame): The original dataset with categorical features.
        
        Returns:
            pd.DataFrame: The converted dataset with unique numerical representations for categorical features.
        """
        # Create a copy to avoid modifying the original dataset
        self.converted_X_encoded = self.X_encoded.copy()

        # Step 4a: Compute the primary frequency-based mapping
        self.compute_primary_frequency_mapping()
    
        # Step 4b: Compute the adaptive Delta r for each categorical feature
        self.compute_adaptive_delta_r()
        
        # Step 4c: Identify and resolve ties by assigning unique r'_jl values
        self.compute_hierarchical_mapping()
        
        # Iterate over each categorical feature
        for col in self.cat_cols:
            hierarchical_mapping = self.hierarchical_mappings[col]
            # Map each category to its r'_jl value
            self.converted_X_encoded[col] = self.converted_X_encoded[col].map(hierarchical_mapping)

        # Store the lookup table and hierarchical mapping for future using a Path()
        self.save_mappings()
        
        
        return self.converted_X_encoded
    

    def get_converted_dataset(self, dataloader=False, test_size=None, random_state=None, batch_size=None):
        """
        Returns the converted dataset with unique numerical representations for categorical features.
        
        """


        if test_size is None:
            test_size = self.test_size
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size

        self.converted_X_encoded = self.Conv()

        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(self.converted_X_encoded, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        
        # Further split the temporary set into validation and test sets
        # val_size_adjusted = self.val_size / (1 - test_size)  # Adjust validation size based on remaining data
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)
        
        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create TensorDatasets for each split
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Return the Datasets for training and test sets
        if dataloader:
            return train_loader, test_loader
        else:
            return train_dataset, test_dataset
    

    def save_mappings(self, directory='mappings/ACI'):
        """
        Saves the hierarchical mappings and lookup tables to pickle files within the specified directory.
        
        Args:
            directory (str or Path, optional): The directory where mapping files will be saved.
                                               Defaults to 'mappings/ACI'.
        """
        # Convert directory to Path object
        path = Path(directory)
        
        # Create the directory if it does not exist
        path.mkdir(parents=True, exist_ok=True)
        
        # Define file paths for hierarchical mappings and lookup tables
        hierarchical_mappings_path = path / 'hierarchical_mappings.pkl'
        lookup_tables_path = path / 'lookup_tables.pkl'
        
        # Serialize the hierarchical mappings to pickle and save
        with open(hierarchical_mappings_path, 'wb') as f:
            pickle.dump(self.hierarchical_mappings, f)
        
        # Serialize the lookup tables to pickle and save
        with open(lookup_tables_path, 'wb') as f:
            pickle.dump(self.lookup_tables, f)
        
        print(f"Hierarchical mappings and lookup tables have been saved to '{path.resolve()}'.")

    def load_mappings(self, directory='mappings/ACI'):
        """
        Loads the hierarchical mappings and lookup tables from pickle files within the specified directory.
        
        Args:
            directory (str or Path, optional): The directory from where mapping files will be loaded.
                                               Defaults to 'mappings/ACI'.
        
        Raises:
            FileNotFoundError: If the mapping files are not found in the specified directory.
        """
        # Convert directory to Path object
        path = Path(directory)
        
        # Define file paths for hierarchical mappings and lookup tables
        hierarchical_mappings_path = path / 'hierarchical_mappings.pkl'
        lookup_tables_path = path / 'lookup_tables.pkl'
        
        # Check if both files exist
        if not hierarchical_mappings_path.exists() or not lookup_tables_path.exists():
            raise FileNotFoundError(f"Mapping files not found in directory '{path.resolve()}'. Please save mappings first.")
        
        # Load hierarchical mappings from pickle
        with open(hierarchical_mappings_path, 'rb') as f:
            self.hierarchical_mappings = pickle.load(f)
        
        # Load lookup tables from pickle
        with open(lookup_tables_path, 'rb') as f:
            self.lookup_tables = pickle.load(f)
        
        print(f"Hierarchical mappings and lookup tables have been loaded from '{path.resolve()}'.")



    def _get_dataset_data(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features and labels from the TensorDataset and converts them into appropriate numpy arrays.

        Parameters:
        dataset (TensorDataset): PyTorch TensorDataset object containing the data.

        Returns:
        X (numpy array): Features from the TensorDataset.
        y (numpy array): Labels from the TensorDataset.
        """
        X_tensor, y_tensor = dataset.tensors

        X = X_tensor.cpu().numpy()
        y = y_tensor.cpu().numpy()


        return X, y
    

    def round_rjl(self, X:torch.Tensor) -> torch.Tensor:
        """
        Rounds the r_jl values of categorical features to the nearest valid r_jl in the lookup tables.
        
        For each categorical feature in X, this method finds the closest r_jl value from the lookup_tables
        and assigns it to the feature. This ensures that all crafted samples have valid r_jl values.
        
        Args:
            X (torch.Tensor): Input tensor with shape (batch_size, d).
        
        Returns:
            torch.Tensor: Tensor with rounded r_jl values for categorical features.
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input X must be a torch.Tensor.")
        
        # Clone X to avoid modifying the original tensor
        X_rounded = X.clone().detach().cpu().numpy()
        
        # Iterate over each categorical feature
        for col in self.cat_cols:
            # Get the column index
            try:
                col_idx = self.feature_names.index(col)
            except ValueError:
                raise ValueError(f"Categorical feature '{col}' not found in feature names.")
            
            # Get the list of valid r_jl values sorted
            rjl_values = np.array(sorted(self.lookup_tables[col].keys()))
            
            # Get the current column values
            feature_values = X_rounded[:, col_idx]
            
            # Find the nearest r_jl for each value
            # Compute the absolute differences
            diff = np.abs(feature_values[:, np.newaxis] - rjl_values[np.newaxis, :])
            
            # Find the index of the smallest difference
            closest_indices = np.argmin(diff, axis=1)
            
            # Get the closest r_jl values
            closest_rjl = rjl_values[closest_indices]
            
            # Update the feature column with the closest r_jl
            X_rounded[:, col_idx] = closest_rjl
        
        # Convert back to torch.Tensor and move to device
        X_rounded_tensor = torch.tensor(X_rounded, dtype=torch.float32)
        
        return X_rounded_tensor
    

    def Revert(self, converted_dataset: TensorDataset, FTT=False) -> TensorDataset:
        """
        Reverts the converted numerical values back to their original categorical values.

        Args:
            converted_dataset (TensorDataset): The dataset with converted numerical categorical features.

        Returns:
            TensorDataset: The dataset with original categorical features restored.
        
        Raises:
            ValueError: If a numerical value does not have a corresponding key in the lookup table.
        """
        # Extract features and labels from the TensorDataset
        X_tensor, y_tensor = converted_dataset.tensors

        # Convert to numpy for easier manipulation
        X_np = X_tensor.cpu().numpy()

        # Iterate over each categorical feature
        for col in self.cat_cols:
            # Get the column index
            try:
                col_idx = self.feature_names.index(col)
            except ValueError:
                raise ValueError(f"Categorical feature '{col}' not found in feature names.")

            # Get the current column values
            feature_values = X_np[:, col_idx]

            # Revert each value using the lookup table
            for i, r_prime in enumerate(feature_values):
                # Round r_prime to the largest p + 1 decimal places to match the lookup table
                r_prime_rounded = round(float(r_prime), self.largest_p + 1)  # Ensure consistent rounding

                # Retrieve the original category using the lookup table
                category = self.lookup_tables[col].get(r_prime_rounded, None)

                if category is not None:
                    X_np[i, col_idx] = category
                else:
                    # Raise an error if the r_prime value is not found in the lookup table
                    available_keys = list(self.lookup_tables[col].keys())
                    raise ValueError(
                        f"Value {r_prime_rounded} for feature '{col}' not found in lookup table. "
                        f"Available keys: {available_keys}"
                    )

        # Convert back to torch.Tensor
        X_reverted_tensor = torch.tensor(X_np, dtype=torch.float32)

        # Return the reverted dataset as a TensorDataset

        if FTT:
            # For using this reverted dataset on FTTransformer, we need to seperate the categorical features and numerical features
            X_categorical = X_reverted_tensor[:, self.cat_cols_idx].to(torch.long)
            X_numerical = X_reverted_tensor[:, self.num_cols_idx]
            return TensorDataset(X_categorical, X_numerical, y_tensor)      
        else:
            
            return TensorDataset(X_reverted_tensor, y_tensor)



    def get_converted_converted_dataset_resampled(self, dataloader=False, test_size=None, random_state=None, batch_size=None, fraction=1.0):

        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(self.converted_X_encoded, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        X_test_upsampled = X_test.copy()

        # Step 1: Calculate 25% of all features to work with
        # This ensures we're only modifying a subset of features for upsampling
        total_features = len(self.feature_names)
        n_features = int(0.25 * total_features)

        # Step 2: Randomly sample features from all available features (both numerical and categorical)
        # This approach ensures unbiased random selection across all feature types
        all_features = self.feature_names
        
        # Ensure we don't select more features than available in the dataset
        n_features = min(n_features, len(all_features))
        
        

        # Step 7: Randomly select 30% of test samples for upsampling
        # This percentage ensures significant impact while maintaining test set integrity
        n_test_samples = X_test_upsampled.shape[0]
        n_samples_to_modify = int(fraction * n_test_samples)
        
        # Get random indices for samples to modify - ensures unbiased sample selection
        samples_to_modify = random.sample(range(n_test_samples), n_samples_to_modify)
    
        print(f"Total test samples: {n_test_samples}")
        print(f"Samples to modify ({fraction*100}%): {n_samples_to_modify}")

        
        # iterate over X_test_upsampled, for the rows that are in samples_to_modify, we will modify the values of the selected features. we resample a value from the lookup table for each selected
        # categorical feature using dirichlet distribution. Then, replace the original value with the resampled value.
        for i in samples_to_modify:

            # Randomly select n_features from all available features
            selected_features = random.sample(all_features, n_features)
            
            # print(f"Total features: {total_features}")
            # print(f"25% of features (n): {n_features}")
            # print(f"Randomly selected features: {selected_features}")
            
            # Step 3: Identify which selected features are numerical and which are categorical
            # Separate the selected features based on their type for further processing
            selected_numeric_features = [feat for feat in selected_features if feat in self.num_cols]
            selected_categorical_features = [feat for feat in selected_features if feat in self.cat_cols]
            
            # print(f"Selected numerical features: {selected_numeric_features}")
            # print(f"Selected categorical features: {selected_categorical_features}")
            # print(f"Number of numerical features selected: {len(selected_numeric_features)}")
            # print(f"Number of categorical features selected: {len(selected_categorical_features)}")

            # total number of selected categorical features and numerical features should match n_features
            assert len(selected_numeric_features) + len(selected_categorical_features) == n_features

            if len(selected_categorical_features) > 0:

                for col in selected_categorical_features:
                    # Get the column index
                    frequency_mappings = self.hierarchical_mappings[col]
                    # get the values all the keys in the frequency_mappings
                    values = list(frequency_mappings.values())

                    # resample a value from the lookup table for each selected
                    # categorical feature using dirichlet distribution. Then, replace the original value with the resampled value.
                    resampled_value = random.choice(values)
                    # Use .iloc for positional indexing: row i, column by name
                    X_test_upsampled.iloc[i, X_test_upsampled.columns.get_loc(col)] = resampled_value                   

            if len(selected_numeric_features) > 0:

                for col in selected_numeric_features:
                    # choose a random value between max and min of the selected column in self.converted_X_encoded
                    max_value = self.converted_X_encoded[col].max()
                    min_value = self.converted_X_encoded[col].min()
                    resampled_value = random.uniform(min_value, max_value)
                    # Use .iloc for positional indexing: row i, column by name
                    X_test_upsampled.iloc[i, X_test_upsampled.columns.get_loc(col)] = resampled_value

        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        X_test_upsampled_tensor = torch.tensor(X_test_upsampled.values, dtype=torch.float32)
        y_test_upsampled_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_upsampled_dataset = TensorDataset(X_test_upsampled_tensor, y_test_upsampled_tensor)

        return train_dataset, test_dataset, test_upsampled_dataset

                
                
    def get_concept_drift_dataset(self, dataloader=False, test_size=None, random_state=None, batch_size=None, train_fraction=0.25, test_fraction=1.0):


        # Split the data into train and temporary sets (temporary set will be further split into validation and test)
        X_train, X_test, y_train, y_test = train_test_split(self.converted_X_encoded, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        X_test_drifted = X_test.copy()
        X_train_drifted = X_train.copy()

        # Step 1: Calculate 25% of all features to work with
        # This ensures we're only modifying a subset of features for upsampling
        total_features = len(self.feature_names)
        n_features = int(0.25 * total_features)

        # Step 2: Randomly sample features from all available features (both numerical and categorical)
        # This approach ensures unbiased random selection across all feature types
        all_features = self.feature_names
        
        # Ensure we don't select more features than available in the dataset
        n_features = min(n_features, len(all_features))


        # Randomly select n_features from all available features
        selected_features = random.sample(all_features, n_features)
        
        # print(f"Total features: {total_features}")
        # print(f"25% of features (n): {n_features}")
        # print(f"Randomly selected features: {selected_features}")
        
        # Step 3: Identify which selected features are numerical and which are categorical
        # Separate the selected features based on their type for further processing
        selected_numeric_features = [feat for feat in selected_features if feat in self.num_cols]
        selected_categorical_features = [feat for feat in selected_features if feat in self.cat_cols]
        
        # print(f"Selected numerical features: {selected_numeric_features}")
        # print(f"Selected categorical features: {selected_categorical_features}")
        # print(f"Number of numerical features selected: {len(selected_numeric_features)}")
        # print(f"Number of categorical features selected: {len(selected_categorical_features)}")

        # total number of selected categorical features and numerical features should match n_features
        assert len(selected_numeric_features) + len(selected_categorical_features) == n_features

        resampled_categorical_values = {}

        if len(selected_categorical_features) > 0:

            for col in selected_categorical_features:
                # Get the column index
                frequency_mappings = self.hierarchical_mappings[col]
                # get the values all the keys in the frequency_mappings
                values = list(frequency_mappings.values())

                # resample a value from the lookup table for each selected
                # categorical feature using dirichlet distribution. Then, replace the original value with the resampled value.
                resampled_value = max(values)
                resampled_categorical_values[col] = resampled_value
        
        resampled_numeric_values = {}

        if len(selected_numeric_features) > 0:

            for col in selected_numeric_features:
                # Calculate mean and standard deviation for the selected numerical column
                # This provides statistical properties of the feature distribution
                mean_value = self.converted_X_encoded[col].mean()
                std_value = self.converted_X_encoded[col].std()


                # Randomly choose fractions between 10% and 30% for mean shift and scaling
                frac_mean = random.uniform(0.1, 0.3)   # Fraction for mean shift
                frac_scale = random.uniform(0.1, 0.3)  # Fraction for scale change
                
                # Calculate transformation parameters
                mean_shift = frac_mean * std_value      # Shift amount based on training std
                scale_std = 1.0 + frac_scale   
                resampled_numeric_values[col] = {'mean_shift': mean_shift, 'scale_std': scale_std}       # Scale factor (> 1 increases variance)

        

        # Step 7: Randomly select test_fraction% of test samples for upsampling
        # This percentage ensures significant impact while maintaining test set integrity
        n_test_samples = X_test_drifted.shape[0]
        n_samples_to_modify = int(test_fraction * n_test_samples)
        
        # Get random indices for samples to modify - ensures unbiased sample selection
        samples_to_modify = random.sample(range(n_test_samples), n_samples_to_modify)
    
        print(f"Total test samples: {n_test_samples}")
        print(f"Samples to modify ({test_fraction*100}%): {n_samples_to_modify}")

        
        # iterate over X_test_upsampled, for the rows that are in samples_to_modify, we will modify the values of the selected features. we resample a value from the lookup table for each selected
        # categorical feature using dirichlet distribution. Then, replace the original value with the resampled value.
        for i in samples_to_modify:

            if len(selected_categorical_features) > 0:

                for col in selected_categorical_features:
                    X_test_drifted.iloc[i, X_test_drifted.columns.get_loc(col)] = resampled_categorical_values[col]
                    
            if len(selected_numeric_features) > 0:

                for col in selected_numeric_features:
                    original_value = X_test_drifted.iloc[i, X_test_drifted.columns.get_loc(col)]
                    mean_shift = resampled_numeric_values[col]['mean_shift']    
                    scale_std = resampled_numeric_values[col]['scale_std']
                    X_test_drifted.iloc[i, X_test_drifted.columns.get_loc(col)] = (original_value + mean_shift) * scale_std

                    # print(f"Original value: {original_value}, Mean shift: {mean_shift}, Scale std: {scale_std}, Transformed value: {X_test_drifted.iloc[i, X_test_drifted.columns.get_loc(col)]}")
                    


        # Randomly select train_fraction% of train samples for upsampling
        n_train_samples = X_train_drifted.shape[0]
        n_samples_to_modify = int(train_fraction * n_train_samples)
        
        # Get random indices for samples to modify - ensures unbiased sample selection
        samples_to_modify = random.sample(range(n_train_samples), n_samples_to_modify)

        print(f"Total train samples: {n_train_samples}")
        print(f"Samples to modify ({train_fraction*100}%): {n_samples_to_modify}")

        for i in samples_to_modify:

            if len(selected_categorical_features) > 0:

                for col in selected_categorical_features:   
                    X_train_drifted.iloc[i, X_train_drifted.columns.get_loc(col)] = resampled_categorical_values[col]
                    
            if len(selected_numeric_features) > 0:

                for col in selected_numeric_features:
                    original_value = X_train_drifted.iloc[i, X_train_drifted.columns.get_loc(col)]
                    mean_shift = resampled_numeric_values[col]['mean_shift']    
                    scale_std = resampled_numeric_values[col]['scale_std']
                    X_train_drifted.iloc[i, X_train_drifted.columns.get_loc(col)] = (original_value + mean_shift) * scale_std


        # Convert the data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        # X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        # y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        X_test_drifted_tensor = torch.tensor(X_test_drifted.values, dtype=torch.float32)
        y_test_drifted_tensor = torch.tensor(y_test, dtype=torch.long)

        X_train_drifted_tensor = torch.tensor(X_train_drifted.values, dtype=torch.float32)
        y_train_drifted_tensor = torch.tensor(y_train, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_drifted_dataset = TensorDataset(X_train_drifted_tensor, y_train_drifted_tensor)
        test_drifted_dataset = TensorDataset(X_test_drifted_tensor, y_test_drifted_tensor)

        return train_dataset, test_dataset, train_drifted_dataset, test_drifted_dataset
            

                              

            


# if __name__ == "__main__":

#     aci = ACI()
#     aci.get_converted_dataset()
    # aci.get_converted_converted_dataset_resampled()
    # aci.get_concept_drift_dataset()

#     # Print the total number of samples in the dataset
#     print(f"Total number of samples in the dataset: {len(aci.X_original)}")

#     # Use np.unique to count the number of samples in each class
#     unique, counts = np.unique(aci.y, return_counts=True)
#     class_counts = dict(zip(unique, counts))
#     print(f"Number of samples in each class: {class_counts}")

#     # Print the number of numerical features
#     print(f"Number of numerical features: {len(aci.num_cols)}")

#     # Print the number of categorical features
#     print(f"Number of categorical features: {len(aci.cat_cols)}")

