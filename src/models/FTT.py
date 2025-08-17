import torch
import torch.nn as nn
from tab_transformer_pytorch import FTTransformer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
from einops import repeat



class FTTModel:
    def __init__(self, data_obj, args=None):
        """
        Initializes a TabNet classifier model with customizable hyperparameters.
        
        Parameters:
        n_d (int): Dimension of the decision prediction layer.
        n_a (int): Dimension of the attention embedding.
        n_steps (int): Number of steps in the architecture.
        gamma (float): Relaxation parameter for TabNet architecture.
        n_independent (int): Number of independent Gated Linear Units layers.
        n_shared (int): Number of shared Gated Linear Units layers.
        momentum (float): Momentum for the batch normalization.
        

        """
        self.model_name = "FTTransformer"
        self.data_obj = data_obj
        self.epochs = 65
        if args.dataset_name.lower() == "higgs":
            self.epochs = 40
        self.device = None
        self.num_workers = args.num_workers if args else 0
        self.batch_size = args.train_batch_size if args else 1024





        # Initialize the FTTransformer model
        self.model_original = FTTransformer(categories=data_obj.FTT_n_categories,
                                   num_continuous=len(data_obj.num_cols),
                                   dim=128,
                                   depth=6,
                                   heads=8,
                                   dim_head=16,
                                   dim_out=data_obj.num_classes,
                                   num_special_tokens=2,
                                   attn_dropout=0.1,
                                   ff_dropout=0.1,
                                   )

        self.model_converted = FTTransformer(categories=(),
                                   num_continuous=len(data_obj.feature_names),
                                   dim=128,
                                   depth=6,
                                   heads=8,
                                   dim_head=16,
                                   dim_out=data_obj.num_classes,
                                   num_special_tokens=1,
                                   attn_dropout=0.1,
                                   ff_dropout=0.1,
                                   )



    def to(self, device: torch.device, model_type: str = "converted"):
        """
        Moves the model to the specified device.

        Parameters:
        device (torch.device): The device to move the model to.
        model_type (str): The type of model to move to the device.
        """
        if model_type == "converted":
            self.model_converted.to(device)  # Set the device for the FTTransformer model
        elif model_type == "original":
            self.model_original.to(device)  # Set the device for the FTTransformer model    
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")
        self.device = device

    

    def get_model(self, model_type: str = "converted"):
        """
        Returns the initialized FTTransformer model for use in training or evaluation.

        Returns:
        model (FTTransformer): The FTTransformer instance.
        """
        if model_type == "converted":
            return self.model_converted
        elif model_type == "original":
            return self.model_original
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")

    def fit(self, train_dataset, val_dataset):
        """
        Trains the FTTransformer model using the provided training data.
        
        Parameters:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        
        Returns:
        None
        """
        # Train the model
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=pin_memory,
                                persistent_workers=self.num_workers > 0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=pin_memory,
                              persistent_workers=self.num_workers > 0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="original")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model_original.parameters(), lr=3.762989816330166e-05, weight_decay=0.0001239780004929955)

        # 11. Training loop with Early Stopping
        epochs = self.epochs
        best_val_loss = float('inf')
        patience = 15
        trigger_times = 0

        best_fttransformer_original = None

        for epoch in range(1, epochs + 1):
            self.model_original.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for X_c, X_n, y_batch in train_loader:
                X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)
                

                optimizer.zero_grad()
                outputs = self.model_original(X_c, X_n)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X_c.size(0)
                
                 
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
            
            avg_train_loss = total_loss / total
            train_accuracy = correct / total
            
            # Validation
            self.model_original.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for X_c, X_n, y_batch in val_loader:
                    X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)

                    outputs = self.model_original(X_c, X_n)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_c.size(0)


                    _, predicted = torch.max(outputs, 1)


                    correct_val += (predicted == y_batch).sum().item()
                    total_val += y_batch.size(0)
            
            avg_val_loss = val_loss / total_val
            val_accuracy = correct_val / total_val
            
            print(f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
                best_fttransformer_original = self.model_original.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

        # 12. Load the best model and evaluate
        self.model_original.load_state_dict(best_fttransformer_original)
        self.model_original.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_c, X_n, y_batch in val_loader:
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_original(X_c, X_n)

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))





    def fit_converted(self, train_dataset, val_dataset):
        """
        Trains the FTTransformer model using the provided training data.
        
        Parameters:
        
        Returns:
        None
        """
        # Train the model
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, pin_memory=pin_memory,
                                persistent_workers=self.num_workers > 0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=pin_memory,
                              persistent_workers=self.num_workers > 0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="converted")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model_converted.parameters(), lr=3.762989816330166e-05, weight_decay=0.0001239780004929955)

        # 11. Training loop with Early Stopping
        epochs = self.epochs
        best_val_loss = float('inf')
        patience = 15
        trigger_times = 0

        best_fttransformer_converted = None

        for epoch in range(1, epochs + 1):
            self.model_converted.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for X_n, y_batch in train_loader:
                X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)
               
                optimizer.zero_grad()
                outputs = self.model_converted(X_c, X_n)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X_c.size(0)
                
 
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
            
            avg_train_loss = total_loss / total
            train_accuracy = correct / total
            
            # Validation
            self.model_converted.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for X_n, y_batch in val_loader:
                    X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                    X_c, X_n, y_batch = X_c.to(device), X_n.to(device), y_batch.to(device)

                    outputs = self.model_converted(X_c, X_n)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_c.size(0)
                    

                    _, predicted = torch.max(outputs, 1)

                    correct_val += (predicted == y_batch).sum().item()
                    total_val += y_batch.size(0)
            
            avg_val_loss = val_loss / total_val
            val_accuracy = correct_val / total_val
            
            print(f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
                best_fttransformer_converted = self.model_converted.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping!")
                    break

        # 12. Load the best model and evaluate
        self.model_converted.load_state_dict(best_fttransformer_converted)
        self.model_converted.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_n, y_batch in val_loader:
                X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_converted(X_c, X_n)

 
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))

    
    def predict(self, X_test):
        """
        Makes predictions on the provided test data using the trained TabNet model.
        
        Parameters:
        X_test (array-like): Input features for testing.
        
        Returns:
        accuracy (float): Accuracy of the model on the test data.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="original")
        val_loader = DataLoader(X_test, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=self.num_workers > 0)
        self.model_original.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_c, X_n, y_batch in val_loader:
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_original(X_c, X_n)

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        
        # Convert lists to NumPy arrays for element-wise comparison
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))
        
        # calculate accuracy
        accuracy = (all_preds == all_targets).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def predict_converted(self, X_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device, model_type="converted")
        val_loader = DataLoader(X_test, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=self.num_workers > 0)
        self.model_converted.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_n, y_batch in val_loader:
                X_c = torch.empty(X_n.shape[0], 0, dtype=torch.long)
                X_c, X_n = X_c.to(device), X_n.to(device)

                outputs = self.model_converted(X_c, X_n)

 
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        # Convert lists to NumPy arrays for element-wise comparison
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        print("Classification Report:")
        print(classification_report(all_targets, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))

        # Calculate accuracy
        accuracy = (all_preds == all_targets).mean()
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
        
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Return logits for the input features

        X_c = torch.empty(X.shape[0], 0, dtype=torch.long)
        X_c = X_c.to(self.device)
        X_n = X

        proba = self.model_converted(X_c, X_n)
        return proba
    
    def forward_original(self, X_c: torch.Tensor, X_n: torch.Tensor) -> torch.Tensor:
        proba = self.model_original(X_c, X_n)
        return proba
    
    def forward_embeddings(self, X_c: torch.Tensor, X_n: torch.Tensor) -> torch.Tensor:
        """
        Returns the penultimate representation from FTTransformer by replicating the
        forward pass but skipping the final linear layer in `ftt_model.to_logits`.

        Steps:
        1) Embedding of categorical (X_c) and continuous (X_n) features.
        2) Concatenation + prepend CLS token.
        3) Transformer forward pass.
        4) Take the CLS token representation (x[:, 0, :]).
        5) Apply the first two layers of `ftt_model.to_logits` (LayerNorm + ReLU),
            but skip the final linear layer. This yields the penultimate embedding.
        
        Args:
        ftt_model (FTTransformer): A trained FTTransformer instance.
        X_c (torch.Tensor): Categorical input of shape (B, num_categories).
        X_n (torch.Tensor): Continuous input of shape (B, num_continuous).
        
        Returns:
        torch.Tensor of shape (B, dim): The penultimate-layer representation.
        """

        x_categ = X_c
        x_numer = X_n

        assert x_categ.shape[-1] == self.model_original.num_categories, f'you must pass in {self.model_original.num_categories} values for your categories input'

        xs = []
        if self.model_original.num_unique_categories > 0:
            x_categ = x_categ + self.model_original.categories_offset

            x_categ = self.model_original.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.model_original.num_continuous > 0:
            x_numer = self.model_original.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.model_original.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = self.model_original.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # ---------------------------------------------------------------
        # 5) Penultimate representation: skip the final linear,
        #    but replicate LN -> ReLU from to_logits if you want.
        # ---------------------------------------------------------------
        # ftt_model.to_logits = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.ReLU(),
        #     nn.Linear(dim, dim_out)
        # )
        # 
        # => The final linear is the last step. We apply LN + ReLU
        #    for the penultimate representation.

        # The first two layers: [0] = LayerNorm(dim), [1] = ReLU()
        ln = self.model_original.to_logits[0]
        act = self.model_original.to_logits[1]

        # Apply LN + ReLU => shape (B, dim)
        penultimate = act(ln(x))

        # Return the penultimate embedding
        return penultimate
    

    def forward_clstokens(self, X_c: torch.Tensor, X_n: torch.Tensor, model_type: str) -> torch.Tensor:
        """
        Returns the penultimate representation from FTTransformer by replicating the
        forward pass but skipping the final linear layer in `ftt_model.to_logits`.
        
        
        

        Steps:
        1) Embedding of categorical (X_c) and continuous (X_n) features.
        2) Concatenation + prepend CLS token.
        3) Transformer forward pass.
        4) Take the CLS token representation (x[:, 0, :]).
        5) Apply the first two layers of `ftt_model.to_logits` (LayerNorm + ReLU),
            but skip the final linear layer. This yields the penultimate embedding.
        
        Args:
        ftt_model (FTTransformer): A trained FTTransformer instance.
        X_c (torch.Tensor): Categorical input of shape (B, num_categories).
        X_n (torch.Tensor): Continuous input of shape (B, num_continuous).
        
        Returns:
        torch.Tensor of shape (B, dim): The penultimate-layer representation.
        """

        if model_type == "original":
            model = self.model_original
        elif model_type == "converted":
            model = self.model_converted
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")

        x_categ = X_c
        x_numer = X_n

        assert x_categ.shape[-1] == model.num_categories, f'you must pass in {model.num_categories} values for your categories input'

        xs = []
        if model.num_unique_categories > 0:
            x_categ = x_categ + model.categories_offset

            x_categ = model.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if model.num_continuous > 0:
            x_numer = model.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(model.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        x, attns = model.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        return x

        
    def save_model(self, filepath, model_type: str = "converted"):
        """
        Saves the trained TabNet model to the specified file path.
        
        Parameters:
        filepath (str): File path to save the model.
        
        Returns:
        None
        """
        if model_type == "converted":
            torch.save(self.model_converted.state_dict(), filepath)
        elif model_type == "original":
            torch.save(self.model_original.state_dict(), filepath)
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")
    
    def load_model(self, filepath, model_type: str = "converted"):
        """
        Loads a pre-trained TabNet model from the specified file path.
        
        Parameters:
        filepath (str): File path to load the model from.
        
        Returns:
        None
        """
        if model_type == "converted":
            self.model_converted.load_state_dict(torch.load(filepath))
        elif model_type == "original":
            self.model_original.load_state_dict(torch.load(filepath))
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")

    
    def eval(self, model_type: str = "converted"):
        if model_type == "converted":
            self.model_converted.eval()
        elif model_type == "original":
            self.model_original.eval()
        else:
            raise ValueError(f"Invalid model type: {model_type}. Please choose 'converted' or 'original'.")



# Example usage:
# Initialize a TabNet model
# tabnet_model = TabNetModel(input_dim=54, output_dim=7)
# Train the model
# tabnet_model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)
# Get the trained model
# model = tabnet_model.get_model()



# if __name__ == "__main__":
#     from CovType import CovType
#     data_obj = CovType()
#     model = FTTModel(data_obj)
#     dataset = data_obj.get_normal_datasets_FTT()
#     model.fit(dataset[0], dataset[1])
