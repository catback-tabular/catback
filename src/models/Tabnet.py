import torch
from pytorch_tabnet.tab_model import TabNetClassifier


class TabNetModel:
    def __init__(self, n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2, momentum=0.3, mask_type='entmax'):
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
        self.model_name = "TabNet"
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.momentum = momentum
        self.mask_type = mask_type

        self.max_epochs = 65



        # Initialize the TabNet model
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            momentum=self.momentum,
            mask_type=self.mask_type

        )


    def to(self, device: torch.device):
        """
        Moves the model to the specified device.

        Parameters:
        device (torch.device): The device to move the model to.
        """
        self.model.device = device  # Set the device for the TabNet model

    

    def get_model(self):
        """
        Returns the initialized TabNet model for use in training or evaluation.
        
        Returns:
        model (TabNetClassifier): The TabNet classifier instance.
        """
        return self.model

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, max_epochs=None, patience=65, batch_size=1024, virtual_batch_size=128):
        """
        Trains the TabNet model using the provided training data.
        
        Parameters:
        X_train (array-like): Input features for training.
        y_train (array-like): Labels for training.
        X_valid (array-like): Input features for validation (optional).
        y_valid (array-like): Labels for validation (optional).
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Number of epochs with no improvement before stopping training early.
        batch_size (int): Batch size for training.
        virtual_batch_size (int): Size of virtual batches for memory-efficient training.
        
        Returns:
        None
        """

        if max_epochs is None:
            max_epochs = self.max_epochs

        # Train the model
        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)] if X_valid is not None and y_valid is not None else None,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size
        )
    
    def predict(self, X_test):
        """
        Makes predictions on the provided test data using the trained TabNet model.
        
        Parameters:
        X_test (array-like): Input features for testing.
        
        Returns:
        preds (array-like): Predicted labels for the test data.
        """
        return self.model.predict(X_test)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Return logits for the input features
        # Convert numpy array to PyTorch tensor
        proba = self.model.predict_proba(X)
        return torch.tensor(proba)
    

    def forward_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the penultimate-layer embeddings from TabNet, i.e., the final
        representation produced by self.model.network(...) before the classification head.

        This can be used to perform tasks like Spectral Signature defense, where we need
        hidden representations rather than final predictions.

        Parameters:
        X (torch.Tensor): Input tensor of shape [batch_size, n_features].

        Returns:
        torch.Tensor: Penultimate-layer representation of shape [batch_size, n_d].
        """
        # According to the PyTorch TabNet library, calling `self.model.network(...)`
        # yields (output, M_loss) where 'output' is a [batch_size, n_d] representation
        # (pre-classification), and M_loss is a mask loss not relevant here.

        device = X.device
        self.model.network.eval()
        with torch.no_grad():
            # Convert input to float if needed
            # X_float = X.float()
            output, _ = self.model.network(X)
            # 'output' should be [batch_size, n_d], the final TabNet representation
        return output.to(device)
    

    def save_model(self, filepath):
        """
        Saves the trained TabNet model to the specified file path.
        
        Parameters:
        filepath (str): File path to save the model.
        
        Returns:
        None
        """
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """
        Loads a pre-trained TabNet model from the specified file path.
        
        Parameters:
        filepath (str): File path to load the model from.
        
        Returns:
        None
        """
        self.model.load_model(filepath)

    
    def eval(self):
        self.model.network.eval()



# Example usage:
# Initialize a TabNet model
# tabnet_model = TabNetModel(input_dim=54, output_dim=7)
# Train the model
# tabnet_model.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)
# Get the trained model
# model = tabnet_model.get_model()
