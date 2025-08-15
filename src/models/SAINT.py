import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from .saint_lib.prepare import argparse_prepare
from .saint_lib.pretrainmodel import SAINT
from .saint_lib.data_openml import DataSetCatCon
import torch.optim as optim
from .saint_lib.augmentations import embed_data_mask
# utils.py contains helper functions such as counting parameters, calculating classification scores, etc.
from .saint_lib.utils import count_parameters, classification_scores, mean_sq_error


class SAINTModel:

    def __init__(self, data_obj, is_numerical=False, args=None):

        self.model_name = "SAINT"
        self.num_workers = args.num_workers if args else 0

        if data_obj.num_classes == 2:
            task = 'binary'
        elif data_obj.num_classes > 2:
            task = 'multiclass'
        else:
            raise ValueError(f"Invalid number of classes for data object: {data_obj.num_classes}")
        
        self.opt = argparse_prepare()
        self.opt.task = task

        # If the task is 'regression', set opt.dtask to 'reg'; otherwise set to 'clf' for classification
        if self.opt.task == 'regression':
            self.opt.dtask = 'reg'
        else:
            self.opt.dtask = 'clf'

        
        # -------------------
        # HYPERPARAMETER ADJUSTMENTS BASED ON DATA
        # -------------------
        # nfeat is the total number of input features (columns) in X_train
        nfeat = len(data_obj.feature_names)

        # If the dataset has > 100 features, reduce the embedding size and/or batch size for memory constraints
        if nfeat > 100:
            self.opt.embedding_size = min(8, self.opt.embedding_size)
            self.opt.batchsize = min(64, self.opt.batchsize)

        
        # If attention type is not just 'col', modify the training configuration:
        # - reduce transformer depth
        # - reduce attention heads
        # - increase dropout
        # - limit embedding size
        if self.opt.attentiontype != 'col':
            self.opt.transformer_depth = 1
            self.opt.attention_heads = min(4, self.opt.attention_heads)
            self.opt.attention_dropout = 0.8
            self.opt.embedding_size = min(32, self.opt.embedding_size)
            self.opt.ff_dropout = 0.8

        # Print the number of features and batchsize being used
        print(nfeat, self.opt.batchsize)
        # Print the configuration stored in opt
        print(self.opt)

        # # If logging to wandb, store the entire config
        # if self.opt.active_log:
        #     wandb.config.update(self.opt)

        if is_numerical or not data_obj.cat_cols:
            self.cat_dims, self.cat_idxs = [], []           
        else:
            self.cat_dims = data_obj.FTT_n_categories
            self.cat_idxs = data_obj.cat_cols_idx
        
        if is_numerical:
            self.con_idxs = [i for i in range(len(data_obj.feature_names))]
        else:
            self.con_idxs = data_obj.num_cols_idx


        # cat_dims array holds the distinct number of categories for each categorical feature.
        # We prepend a 1 for the [CLS] token (a special token used in transformers).
        self.cat_dims = np.append(np.array([1]), np.array(self.cat_dims)).astype(int)

        # -------------------
        # MODEL INITIALIZATION
        # -------------------
        self.model = SAINT(
            categories=tuple(self.cat_dims),
            num_continuous=len(self.con_idxs),
            dim=self.opt.embedding_size,
            dim_out=1,  # dimension of each feature embedding's output
            depth=self.opt.transformer_depth,
            heads=self.opt.attention_heads,
            attn_dropout=self.opt.attention_dropout,
            ff_dropout=self.opt.ff_dropout,
            mlp_hidden_mults=(4, 2),
            cont_embeddings=self.opt.cont_embeddings,
            attentiontype=self.opt.attentiontype,
            final_mlp_style=self.opt.final_mlp_style,
            y_dim=data_obj.num_classes
                )
                
        
        self.y_dim = data_obj.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def to(self, device: torch.device):
        """
        Moves the model to the specified device.

        Parameters:
        device (torch.device): The device to move the model to.
        """

        self.model.to(device)

    def eval(self):
        self.model.eval()

    
    def get_model(self):
        """
        Returns the initialized SAINT model for use in training or evaluation.
        
        Returns:
        model (SAINT): The SAINT model instance.
        """
        return self.model
    

    def fit(self, X_train, y_train, X_val, y_val):


        # Create a mask array with the same shape as X_train.
        # If the value is NaN, the mask value is 0 and 1 otherwise.
        X_train_mask = (np.isnan(X_train) == False).astype(int)

        # Create a mask array with the same shape as X_val.
        # If the value is NaN, the mask value is 0 and 1 otherwise.
        X_val_mask = (np.isnan(X_val) == False).astype(int)


        # reshape y_train and y_val to be (N, 1) instead of (N,)
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        X_train = {
            'data': X_train,
            'mask': X_train_mask
        }

        X_val = {
            'data': X_val,
            'mask': X_val_mask
        }

        y_train = {
            'data': y_train
        }

        y_val = {
            'data': y_val
        }

        train_ds = DataSetCatCon(X_train, y_train, self.cat_idxs)
        val_ds = DataSetCatCon(X_val, y_val, self.cat_idxs)

        # Wrap them in DataLoaders
        pin_memory = torch.cuda.is_available()
        trainloader = DataLoader(train_ds, batch_size=self.opt.batchsize, shuffle=True,
                                num_workers=self.num_workers, pin_memory=pin_memory,
                                persistent_workers=self.num_workers > 0)
        validloader = DataLoader(val_ds, batch_size=self.opt.batchsize, shuffle=False,
                                num_workers=self.num_workers, pin_memory=pin_memory,
                                persistent_workers=self.num_workers > 0)
                                




        # -------------------
        # OPTIONAL: WANDB LOGGING
        # -------------------
        if self.opt.active_log:
            import wandb
            # If we are doing pretraining, log it accordingly
            if self.opt.pretrain:
                wandb.init(project="saint_v2_all", group=self.opt.run_name,
                        name=f'pretrain_{self.opt.task}_{str(self.opt.attentiontype)}_{str(self.opt.dset_id)}_{str(self.opt.set_seed)}')
            else:
                # For multiclass, log to a slightly different project name; else default
                if self.opt.task == 'multiclass':
                    wandb.init(project="saint_v2_all_kamal", group=self.opt.run_name,
                            name=f'{self.opt.task}_{str(self.opt.attentiontype)}_{str(self.opt.dset_id)}_{str(self.opt.set_seed)}')
                else:
                    wandb.init(project="saint_v2_all", group=self.opt.run_name,
                            name=f'{self.opt.task}_{str(self.opt.attentiontype)}_{str(self.opt.dset_id)}_{str(self.opt.set_seed)}')



        # LOSS CRITERION SETUP
        # -------------------
        # Depending on the task and y_dim, choose an appropriate loss function
        if self.y_dim == 2 and self.opt.task == 'binary':
            # For binary classification with 2 output classes
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.y_dim > 2 and self.opt.task == 'multiclass':
            # For multiclass classification
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.opt.task == 'regression':
            # For regression tasks, use mean-squared error
            criterion = nn.MSELoss().to(self.device)
        else:
            # Raise an error if none of these conditions are met
            raise 'case not written yet'

        # Move the model to the chosen device (GPU if available, else CPU)
        self.model.to(self.device)

        # -------------------
        # OPTIMIZER & SCHEDULER
        # -------------------
        if self.opt.optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.opt.lr,
                                momentum=0.9, weight_decay=5e-4)
            from saint_lib.utils import get_scheduler
            scheduler = get_scheduler(self.opt, optimizer)
        elif self.opt.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr)
        elif self.opt.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr)

        # Initialize variables to keep track of the best validation metrics
        best_valid_auroc = 0
        best_valid_accuracy = 0
        best_test_auroc = 0
        best_test_accuracy = 0
        best_valid_rmse = 100000  # a large initial value for RMSE tracking

        print('Training begins now.')


                
        # -------------------
        # MAIN TRAINING LOOP
        # -------------------
        for epoch in range(self.opt.epochs):
            # Switch the model to training mode
            self.model.train()
            
            # Variable to accumulate the total loss for the epoch
            running_loss = 0.0

            # Loop through each batch from the train DataLoader
            for i, data in enumerate(trainloader, 0):
                # Zero the gradients (PyTorch accumulates gradients)
                optimizer.zero_grad()

                # data[0] = x_categ (categorical features)
                # data[1] = x_cont (continuous features)
                # data[2] = y_gts (ground truth labels)
                # data[3] = cat_mask (mask for categorical features)
                # data[4] = con_mask (mask for continuous features)
                x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(self.device), \
                                                            data[1].to(self.device), \
                                                            data[2].to(self.device), \
                                                            data[3].to(self.device), \
                                                            data[4].to(self.device)
                


                # embed_data_mask converts raw data to embeddings for both categorical and continuous features
                # taking into account any masking that might be required.
                _, x_categ_enc, x_cont_enc = embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, self.model)

                # Pass the encoded embeddings through the Transformer
                reps = self.model.transformer(x_categ_enc, x_cont_enc)

                # reps[:,0,:] corresponds to the [CLS] token's representation, which is often used
                # as a summarization vector for classification or regression
                y_reps = reps[:, 0, :]

                # mlpfory is the final MLP layer that predicts the output (class or reg)
                y_outs = self.model.mlpfory(y_reps)

                # Calculate the loss based on the task
                if self.opt.task == 'regression':
                    loss = criterion(y_outs, y_gts)
                else:
                    # For classification, we typically have class indices in y_gts, so we use squeeze
                    loss = criterion(y_outs, y_gts.squeeze())

                # Backpropagate the loss to compute gradients
                loss.backward()

                # Update parameters
                optimizer.step()

                # If we are using SGD with a scheduler, step the scheduler once per iteration
                if self.opt.optimizer == 'SGD':
                    scheduler.step()

                # Accumulate the loss
                running_loss += loss.item()

            # Optionally log the training loss to wandb
            if self.opt.active_log:
                wandb.log({
                    'epoch': epoch,
                    'train_epoch_loss': running_loss,
                    'loss': loss.item()
                })

            # Every 5 epochs, check performance on validation and test sets
            if epoch % 5 == 0:
                # Switch the model to eval mode for validation
                self.model.eval()
                with torch.no_grad():
                    if self.opt.task in ['binary', 'multiclass']:
                        # classification_scores calculates accuracy and AUROC
                        accuracy, auroc = classification_scores(self.model, validloader, self.device, self.opt.task, vision_dset=False)
                        # test_accuracy, test_auroc = classification_scores(self.model, testloader, self.device, self.opt.task, vision_dset=False)

                        print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                            (epoch + 1, accuracy, auroc))
                        # print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                        #     (epoch + 1, test_accuracy, test_auroc))

                        # Log validation/test metrics if using wandb
                        if self.opt.active_log:
                            wandb.log({'valid_accuracy': accuracy, 'valid_auroc': auroc})
                            # wandb.log({'test_accuracy': test_accuracy, 'test_auroc': test_auroc})

                        # Save the model if it achieves better validation accuracy (and/or AUROC)
                        # For multiclass, we track best accuracy.
                        if self.opt.task == 'multiclass':
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                                # best_test_auroc = test_auroc
                                # best_test_accuracy = test_accuracy
                                # torch.save(self.model.state_dict(), f'{self.modelsave_path}/bestmodel.pth')
                        else:
                            # For binary, track best accuracy as well, or best AUROC if preferred
                            if accuracy > best_valid_accuracy:
                                best_valid_accuracy = accuracy
                                # best_test_auroc = test_auroc
                                # best_test_accuracy = test_accuracy
                                # torch.save(self.model.state_dict(), f'{self.modelsave_path}/bestmodel.pth')

                    else:
                        # For regression, compute RMSE on valid and test sets
                        valid_rmse = mean_sq_error(self.model, validloader, self.device, vision_dset=False)
                        # test_rmse = mean_sq_error(self.model, testloader, self.device, vision_dset=False)

                        print('[EPOCH %d] VALID RMSE: %.3f' %
                            (epoch + 1, valid_rmse))
                        # print('[EPOCH %d] TEST RMSE: %.3f' %
                        #     (epoch + 1, test_rmse))

                        # Log these metrics if wandb is active
                        if self.opt.active_log:
                            wandb.log({'valid_rmse': valid_rmse})
                            # wandb.log({'test_rmse': test_rmse})

                        # If this validation RMSE is better than our previous best, save the model
                        if valid_rmse < best_valid_rmse:
                            best_valid_rmse = valid_rmse
                            # best_test_rmse = test_rmse
                            # torch.save(self.model.state_dict(), f'{self.modelsave_path}/bestmodel.pth')

                # Switch back to train mode after validation
                self.model.train()

        # -------------------
        # AFTER TRAINING COMPLETES
        # -------------------
        # Count total number of parameters in the model for reporting
        total_parameters = count_parameters(self.model)
        print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))

        # # Print out the best results on the test set depending on the task
        # if self.opt.task == 'binary':
        #     print('AUROC on best model:  %.3f' % (best_test_auroc))
        # elif self.opt.task == 'multiclass':
        #     print('Accuracy on best model:  %.3f' % (best_test_accuracy))
        # else:
        #     print('RMSE on best model:  %.3f' % (best_test_rmse))

        # # Also log these final results to wandb if active
        # if opt.active_log:
        #     if opt.task == 'regression':
        #         wandb.log({
        #             'total_parameters': total_parameters,
        #             'test_rmse_bestep': best_test_rmse,
        #             'cat_dims': len(cat_idxs),
        #             'con_dims': len(con_idxs)
        #         })
        #     else:
        #         wandb.log({
        #             'total_parameters': total_parameters,
        #             'test_auroc_bestep': best_test_auroc,
        #             'test_accuracy_bestep': best_test_accuracy,
        #             'cat_dims': len(cat_idxs),
        #             'con_dims': len(con_idxs)
        #         })



    def predict(self, X_test, y_test):


        # Create a mask array with the same shape as X_train.
        # If the value is NaN, the mask value is 0 and 1 otherwise.
        X_test_mask = (np.isnan(X_test) == False).astype(int)

        # reshape y_train and y_val to be (N, 1) instead of (N,)
        y_test = y_test.reshape(-1, 1)


        X_test = {
            'data': X_test,
            'mask': X_test_mask
        }


        y_test = {
            'data': y_test
        }

        test_ds = DataSetCatCon(X_test, y_test, self.cat_idxs)

        # Wrap them in DataLoaders
        pin_memory = torch.cuda.is_available()
        testloader = DataLoader(test_ds, batch_size=self.opt.batchsize, shuffle=False,
                              num_workers=self.num_workers, pin_memory=pin_memory,
                              persistent_workers=self.num_workers > 0)




        # -------------------
        # OPTIONAL: WANDB LOGGING
        # -------------------
        if self.opt.active_log:
            import wandb
            # If we are doing pretraining, log it accordingly
            if self.opt.pretrain:
                wandb.init(project="saint_v2_all", group=self.opt.run_name,
                        name=f'pretrain_{self.opt.task}_{str(self.opt.attentiontype)}_{str(self.opt.dset_id)}_{str(self.opt.set_seed)}')
            else:
                # For multiclass, log to a slightly different project name; else default
                if self.opt.task == 'multiclass':
                    wandb.init(project="saint_v2_all_kamal", group=self.opt.run_name,
                            name=f'{self.opt.task}_{str(self.opt.attentiontype)}_{str(self.opt.dset_id)}_{str(self.opt.set_seed)}')
                else:
                    wandb.init(project="saint_v2_all", group=self.opt.run_name,
                            name=f'{self.opt.task}_{str(self.opt.attentiontype)}_{str(self.opt.dset_id)}_{str(self.opt.set_seed)}')



        # LOSS CRITERION SETUP
        # -------------------
        # Depending on the task and y_dim, choose an appropriate loss function
        if self.y_dim == 2 and self.opt.task == 'binary':
            # For binary classification with 2 output classes
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.y_dim > 2 and self.opt.task == 'multiclass':
            # For multiclass classification
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.opt.task == 'regression':
            # For regression tasks, use mean-squared error
            criterion = nn.MSELoss().to(self.device)
        else:
            # Raise an error if none of these conditions are met
            raise 'case not written yet'

        # Move the model to the chosen device (GPU if available, else CPU)
        self.model.to(self.device)

        # -------------------
        # OPTIMIZER & SCHEDULER
        # -------------------
        if self.opt.optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.opt.lr,
                                momentum=0.9, weight_decay=5e-4)
            from saint_lib.utils import get_scheduler
            scheduler = get_scheduler(self.opt, optimizer)
        elif self.opt.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr)
        elif self.opt.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr)

        # Initialize variables to keep track of the best validation metrics

        best_test_auroc = 0
        best_test_accuracy = 0

        print('Testing begins now.')


        
        # Switch the model to eval mode for validation
        self.model.eval()
        with torch.no_grad():
            if self.opt.task in ['binary', 'multiclass']:
                # classification_scores calculates accuracy and AUROC
                test_accuracy, test_auroc = classification_scores(self.model, testloader, self.device, self.opt.task, vision_dset=False)
                # test_accuracy, test_auroc = classification_scores(self.model, testloader, self.device, self.opt.task, vision_dset=False)

                print('[TEST] ACCURACY: %.3f, AUROC: %.3f' %
                    (test_accuracy, test_auroc))

                # Log validation/test metrics if using wandb
                if self.opt.active_log:
                    wandb.log({'test_accuracy': test_accuracy, 'test_auroc': test_auroc})

                # Save the model if it achieves better validation accuracy (and/or AUROC)
                # For multiclass, we track best accuracy.
                if self.opt.task == 'multiclass':
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy
                        # torch.save(self.model.state_dict(), f'{self.modelsave_path}/bestmodel.pth')
                else:
                    # For binary, track best accuracy as well, or best AUROC if preferred
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        best_test_auroc = test_auroc

            else:
                # For regression, compute RMSE on valid and test sets
                test_rmse = mean_sq_error(self.model, testloader, self.device, vision_dset=False)
                # test_rmse = mean_sq_error(self.model, testloader, self.device, vision_dset=False)

                print('[TEST] RMSE: %.3f' %
                    (test_rmse))

                # Log these metrics if wandb is active
                if self.opt.active_log:
                    wandb.log({'test_rmse': test_rmse})

                # If this validation RMSE is better than our previous best, save the model
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    # best_test_rmse = test_rmse
                    # torch.save(self.model.state_dict(), f'{self.modelsave_path}/bestmodel.pth')



        # -------------------
        # AFTER TRAINING COMPLETES
        # -------------------
        # Count total number of parameters in the model for reporting
        total_parameters = count_parameters(self.model)
        print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))

        # Print out the best results on the test set depending on the task
        if self.opt.task == 'binary':
            print('AUROC on best model:  %.3f' % (best_test_auroc))
        elif self.opt.task == 'multiclass':
            print('Accuracy on best model:  %.3f' % (best_test_accuracy))
        else:
            print('RMSE on best model:  %.3f' % (best_test_rmse))

        # Also log these final results to wandb if active
        if self.opt.active_log:
            if self.opt.task == 'regression':
                wandb.log({
                    'total_parameters': total_parameters,
                    'test_rmse_bestep': best_test_rmse,
                    'cat_dims': len(self.cat_idxs),
                    'con_dims': len(self.con_idxs)
                })
            else:
                wandb.log({
                    'total_parameters': total_parameters,
                    'test_auroc_bestep': best_test_auroc,
                    'test_accuracy_bestep': best_test_accuracy,
                    'cat_dims': len(self.cat_idxs),
                    'con_dims': len(self.con_idxs)
                })

        return best_test_accuracy



    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))


    # def forward(self, X: torch.Tensor) -> torch.Tensor:
    #     ''' Returns logits for the input features'''

    #     # Convert torch tensor to numpy array
    #     X = X.numpy()

    #     X_mask = (np.isnan(X) == False).astype(int)

    #     X_categ, X_cont = X[:, self.cat_idxs], X[:, self.con_idxs]

    #     X_categ_mask = X_mask[:, self.cat_idxs].astype(np.int64)
    #     X_cont_mask = X_mask[:, self.con_idxs].astype(np.int64)

    #     # As a cls token, add a column of 0s to the beginning of X_categ
    #     X_categ = np.concatenate([np.zeros((X_categ.shape[0], 1)), X_categ], axis=1)

    #     # As a cls mask, add a column of 1s to the beginning of X_categ_mask
    #     X_categ_mask = np.concatenate([np.ones((X_categ_mask.shape[0], 1)), X_categ_mask], axis=1)

    #     # convert all the numpy arrays to torch tensors
    #     X_categ = torch.tensor(X_categ, dtype=torch.int64)
    #     X_cont = torch.tensor(X_cont, dtype=torch.float32)
    #     X_categ_mask = torch.tensor(X_categ_mask, dtype=torch.int64)
    #     X_cont_mask = torch.tensor(X_cont_mask, dtype=torch.int64)

    #     self.model.eval()
    #     with torch.no_grad():

    #         _, X_categ_enc, X_cont_enc = embed_data_mask(X_categ, X_cont, X_categ_mask, X_cont_mask, self.model)

    #         reps = self.model.transformer(X_categ_enc, X_cont_enc)

    #         y_reps = reps[:, 0, :]

    #         y_outs = self.model.mlpfory(y_reps)

    #     return y_outs



    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for the input features
        """

        # Create a mask for non-NaN values (1 for valid values, 0 for NaN)
        X_mask = (~X.isnan()).int()


        # Separate categorical and continuous columns
        X_categ = X[:, self.cat_idxs]
        X_cont = X[:, self.con_idxs]

        # Separate the masks for categorical and continuous columns
        X_categ_mask = X_mask[:, self.cat_idxs]
        X_cont_mask = X_mask[:, self.con_idxs]

        # Add a "CLS" token (0) as the first column of X_categ
        cls_token = torch.zeros((X_categ.size(0), 1), device=X.device)
        X_categ = torch.cat([cls_token, X_categ], dim=1)

        # Add a "CLS" mask (1) as the first column of X_categ_mask
        cls_mask = torch.ones((X_categ_mask.size(0), 1), device=X.device)
        X_categ_mask = torch.cat([cls_mask, X_categ_mask], dim=1)

        # Convert to appropriate dtypes
        X_categ = X_categ.long()
        X_cont = X_cont.float()
        X_categ_mask = X_categ_mask.long()
        X_cont_mask = X_cont_mask.long()

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _, X_categ_enc, X_cont_enc = embed_data_mask(
                X_categ, X_cont, X_categ_mask, X_cont_mask, self.model
            )
            reps = self.model.transformer(X_categ_enc, X_cont_enc)
            y_reps = reps[:, 0, :]       # Take [CLS] token representation
            y_outs = self.model.mlpfory(y_reps)

        return y_outs



    def forward_embeddings(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns the penultimate-layer embeddings from SAINT, i.e., the final
        representation produced by self.model.transformer(...) before the classification head.
        """
        # Create a mask for non-NaN values (1 for valid values, 0 for NaN)
        X_mask = (~X.isnan()).int()


        # Separate categorical and continuous columns
        X_categ = X[:, self.cat_idxs]
        X_cont = X[:, self.con_idxs]

        # Separate the masks for categorical and continuous columns
        X_categ_mask = X_mask[:, self.cat_idxs]
        X_cont_mask = X_mask[:, self.con_idxs]

        # Add a "CLS" token (0) as the first column of X_categ
        cls_token = torch.zeros((X_categ.size(0), 1), device=X.device)
        X_categ = torch.cat([cls_token, X_categ], dim=1)

        # Add a "CLS" mask (1) as the first column of X_categ_mask
        cls_mask = torch.ones((X_categ_mask.size(0), 1), device=X.device)
        X_categ_mask = torch.cat([cls_mask, X_categ_mask], dim=1)

        # Convert to appropriate dtypes
        X_categ = X_categ.long()
        X_cont = X_cont.float()
        X_categ_mask = X_categ_mask.long()
        X_cont_mask = X_cont_mask.long()

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _, X_categ_enc, X_cont_enc = embed_data_mask(
                X_categ, X_cont, X_categ_mask, X_cont_mask, self.model
            )
            reps = self.model.transformer(X_categ_enc, X_cont_enc)
            y_reps = reps[:, 0, :]       # Take [CLS] token representation
            # y_outs = self.model.mlpfory(y_reps)

        return y_reps






    def forward_reps(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for the input features
        """

        # Create a mask for non-NaN values (1 for valid values, 0 for NaN)
        X_mask = (~X.isnan()).int()


        # Separate categorical and continuous columns
        X_categ = X[:, self.cat_idxs]
        X_cont = X[:, self.con_idxs]

        # Separate the masks for categorical and continuous columns
        X_categ_mask = X_mask[:, self.cat_idxs]
        X_cont_mask = X_mask[:, self.con_idxs]

        # Add a "CLS" token (0) as the first column of X_categ
        cls_token = torch.zeros((X_categ.size(0), 1), device=X.device)
        X_categ = torch.cat([cls_token, X_categ], dim=1)

        # Add a "CLS" mask (1) as the first column of X_categ_mask
        cls_mask = torch.ones((X_categ_mask.size(0), 1), device=X.device)
        X_categ_mask = torch.cat([cls_mask, X_categ_mask], dim=1)

        # Convert to appropriate dtypes
        X_categ = X_categ.long()
        X_cont = X_cont.float()
        X_categ_mask = X_categ_mask.long()
        X_cont_mask = X_cont_mask.long()

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _, X_categ_enc, X_cont_enc = embed_data_mask(
                X_categ, X_cont, X_categ_mask, X_cont_mask, self.model
            )
            reps = self.model.transformer(X_categ_enc, X_cont_enc)
            # y_reps = reps[:, 0, :]       # Take [CLS] token representation
            # y_outs = self.model.mlpfory(y_reps)

        return reps

