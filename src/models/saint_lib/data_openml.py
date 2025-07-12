import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset

# -------------------------------------------------------------------------
# A small helper function to print elapsed time in a human-readable format.
# -------------------------------------------------------------------------
def simple_lapsed_time(text, lapsed):
    # Break down the total 'lapsed' time (in seconds) into hours, minutes, seconds
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Print the result with the 'text' prefix: "HH:MM:SS.xx"
    print(text + ": {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

# -------------------------------------------------------------------------
# This function returns a dictionary of lists of dataset IDs from OpenML 
# for different tasks: 'binary', 'multiclass', and 'regression'.
# These are used to identify and retrieve OpenML datasets later.
# -------------------------------------------------------------------------
def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487, 44, 1590, 42178, 1111, 31, 42733, 1494, 1017, 4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression': [541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }
    # Return the list corresponding to the given 'task'
    return dataset_ids[task]

# -------------------------------------------------------------------------
# A helper function to combine data (X) and labels (y) into a single dataframe.
#   - X is a dictionary where 'X["data"]' is a NumPy array of feature columns.
#   - y is also a dictionary, with 'y["data"]' storing the labels in the first column.
# We create a single Pandas DataFrame containing all features plus a column named 'target'.
# -------------------------------------------------------------------------
def concat_data(X, y):
    # Convert X['data'] to a DataFrame, and similarly transform y['data'] 
    # so we can easily concatenate them as columns.
    return pd.concat([
        pd.DataFrame(X['data']),
        pd.DataFrame(y['data'][:, 0].tolist(), columns=['target'])
    ], axis=1)

# -------------------------------------------------------------------------
# Splits a dataset into a smaller subset using provided indices, while 
# preserving both feature data and mask information for missing values.
#   - X is a pandas DataFrame of shape [N, D], where N is samples, D is features.
#   - y is the labels array.
#   - nan_mask is a DataFrame of the same shape as X, containing 0s and 1s
#     indicating whether the value is missing (0) or not (1).
#   - indices are the subset of row indices to select for the split (train/valid/test).
# Returns two dictionaries:
#   1) x_d with 'data': selected rows from X, 'mask': selected rows from nan_mask
#   2) y_d with 'data': selected rows from y
# -------------------------------------------------------------------------
def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],     # Subset of the actual data
        'mask': nan_mask.values[indices]  # Subset of the missing-value mask
    }
    
    # Check shape consistency: data and mask should have the same shape
    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)  # Store labels in a 2D array shape: (N, 1)
    }
    
    return x_d, y_d

# -------------------------------------------------------------------------
# data_prep_openml is the main function that:
#   1) Downloads a dataset from OpenML by its dataset ID (ds_id).
#   2) Extracts features (X) and labels (y).
#   3) Distinguishes between categorical and continuous features.
#   4) Handles missing values and label encodings.
#   5) Splits into train/valid/test subsets according to 'datasplit' ratios.
#   6) Computes mean and std for continuous columns in the training set 
#      (for normalization usage later).
#
# It returns the following objects:
#   cat_dims:  list with number of unique categories for each categorical feature
#   cat_idxs:  list of indices in X that are categorical
#   con_idxs:  list of indices in X that are continuous
#   X_train, y_train, X_valid, y_valid, X_test, y_test: 
#       each is a dictionary with 'data' and 'mask'
#   train_mean, train_std: 
#       the mean and std of continuous features from the training split
# -------------------------------------------------------------------------
def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):
    
    # Ensure reproducibility by setting the random seed
    np.random.seed(seed)
    
    # Download the OpenML dataset by its ID
    dataset = openml.datasets.get_dataset(ds_id)
    
    # Retrieve the data and metadata. 'X' is a DataFrame of features,
    # 'y' are the corresponding labels, 'categorical_indicator' tells 
    # which columns in X are categorical, 'attribute_names' are the column names.
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )
    
    # Below are some dataset-specific adjustments/fixes:
    # 1) For dataset 42178, we fix the type of a particular column and remove rows where it's zero.
    if ds_id == 42178:
        categorical_indicator = [True, False, True, True, False, True, True, True, 
                                 True, True, True, True, True, True, True, True, 
                                 True, False, False]
        
        # Some records have a space ' ' which is invalid for numeric column 'TotalCharges'. Replace with '0'.
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp]
        
        # Remove rows where 'TotalCharges' is zero, adjusting X and y correspondingly
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    
    # 2) For certain dataset IDs, reduce dataset size to 50,000 rows (for memory/time efficiency).
    if ds_id in [42728, 42705, 42729, 42571]:
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    
    # Identify column names that are categorical (based on 'categorical_indicator')
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
    # The remaining columns are continuous
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))
    
    # Indices of categorical columns 
    cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
    # Indices of continuous columns
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    # Ensure each categorical column has object (string) type
    for col in categorical_columns:
        X[col] = X[col].astype("object")

    # Add a column "Set" that randomly assigns each row as "train", "valid", or "test"
    X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))

    # Create train, valid, and test subsets based on "Set" column
    train_indices = X[X.Set == "train"].index
    valid_indices = X[X.Set == "valid"].index
    test_indices  = X[X.Set == "test"].index

    # Drop the auxiliary "Set" column, since we've already used it
    X = X.drop(columns=['Set'])
    
    # Replace all remaining NaNs with the string "MissingValue"
    temp = X.fillna("MissingValue")
    # Create a mask where 1 indicates "not missing" and 0 indicates "missing"
    nan_mask = temp.ne("MissingValue").astype(int)
    
    # cat_dims will hold the distinct number of categories for each categorical feature
    cat_dims = []
    for col in categorical_columns:
        # Fill any NaNs with the placeholder "MissingValue"
        X[col] = X[col].fillna("MissingValue")
        
        # Encode the categorical string values as integer class labels
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        
        # Append the number of unique classes to cat_dims
        cat_dims.append(len(l_enc.classes_))

    # For continuous columns, fill any NaNs with the mean of that column (based on training set)
    for col in cont_columns:
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)

    # Convert 'y' to a NumPy array
    y = y.values
    
    # If not a regression task, encode 'y' with LabelEncoder (for classification)
    if task != 'regression':
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)

    # Create split dictionaries: (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test,  y_test  = data_split(X, y, nan_mask, test_indices)

    # Compute mean and std of continuous columns (in the training split) for normalization
    train_mean = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0)
    train_std  = np.array(X_train['data'][:, con_idxs], dtype=np.float32).std(0)
    
    # To avoid division by zero or near-zero, clamp very small std values to 1e-6
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    # Return all relevant objects
    return (cat_dims, cat_idxs, con_idxs, 
            X_train, y_train, X_valid, y_valid, X_test, y_test,
            train_mean, train_std)

# -------------------------------------------------------------------------
# A custom PyTorch dataset class that stores the categorical and continuous 
# features, as well as a mask for missing values. This allows us to handle 
# them later in our model pipeline.
#
#  - X: dictionary with 'data' and 'mask'
#  - Y: dictionary with 'data' for labels
#  - cat_cols: list of indices for categorical columns
#  - task: 'clf' or 'reg' for classification or regression
#  - continuous_mean_std: the (mean, std) arrays for normalizing continuous data
#
# The __getitem__ method returns:
#   1. Concatenation of a zero CLS token with the categorical columns
#   2. The continuous feature values
#   3. The target label
#   4. Concatenation of a mask for CLS token (all 1s) with the categorical mask
#   5. The continuous mask
# -------------------------------------------------------------------------
class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):
        # Convert cat_cols to list, in case it's not
        cat_cols = list(cat_cols)
        
        # Extract the mask and the actual data from X
        X_mask = X['mask'].copy()
        X_data = X['data'].copy()
        
        # Identify the continuous columns as the set difference of all columns minus cat_cols
        con_cols = list(set(np.arange(X_data.shape[1])) - set(cat_cols))
        
        # Categorical data: integer type
        self.X1 = X_data[:, cat_cols].copy().astype(np.int64)
        
        # Continuous data: float32 type
        self.X2 = X_data[:, con_cols].copy().astype(np.float32)
        
        # Masks for categorical and continuous columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)
        
        # Store labels
        if task == 'clf':
            # For classification, keep labels as is (likely int-coded classes)
            self.y = Y['data']
        else:
            # For regression, ensure float32
            self.y = Y['data'].astype(np.float32)
        
        # Create a "CLS token" for each data sample, stored as 0
        self.cls = np.zeros_like(self.y, dtype=int)
        # Create a mask for the CLS token, all set to 1 (meaning not missing)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        
        # If we have mean and std for continuous features, perform standard normalization
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        # Return the total number of samples
        return len(self.y)
    
    def __getitem__(self, idx):
        # Return:
        # 1) Concatenate CLS token with categorical data
        # 2) Continuous data
        # 3) The label
        # 4) Concatenate CLS mask with categorical mask
        # 5) Continuous mask
        return (
            np.concatenate((self.cls[idx], self.X1[idx])),
            self.X2[idx],
            self.y[idx],
            np.concatenate((self.cls_mask[idx], self.X1_mask[idx])),
            self.X2_mask[idx]
        )
