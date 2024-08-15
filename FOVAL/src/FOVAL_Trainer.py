import pickle
import random
from collections import defaultdict
import csv
import os
from matplotlib.gridspec import GridSpec
import keyboard  # Import the keyboard library
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

import wandb
import torch
import scipy.stats as stats
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from FOVAL_Preprocessor import subjective_normalization_dataset, separate_features_and_targets, selected_features
from RobustVision_Dataset import create_features, create_sequences
from Utilities import create_lstm_tensors_dataset, create_dataloaders_dataset, define_model_and_optim, analyzeResiduals, \
    print_results

pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Use this to print the name of the first device
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU
n_epochs = 500


def save_activations_validation(intermediates, target_vector, name, save_dir):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    activations_data = {}  # Dictionary to store activations data

    # Save intermediate activations
    for i, (key, activation) in enumerate(intermediates.items()):
        # Convert the tensor to numpy array and save it
        activations_data[key] = activation.detach().cpu().numpy()

    # Save target vector
    activations_data['target_vector'] = target_vector.detach().cpu().numpy()

    # Save the activations data dictionary
    activations_file = os.path.join(save_dir, f'{name}_activations.npy')
    np.save(activations_file, activations_data)

    return activations_file


class FOVAL_Trainer:
    pd.option_context('mode.use_inf_as_na', True)

    def __init__(self):

        self.input_size = None
        self.train_loader = None
        self.valid_loader = None
        self.dataset_obj = None
        # Define the CSV filename
        self.all_predictions_array = None
        self.all_true_values_array = None
        self.userFolder = ""
        self.fold_results = None
        self.best_test_smae = float('inf')
        self.best_test_mse = float('inf')
        self.best_test_mae = float('inf')
        self.test_features_input = None
        self.test_targets_input = None

        self.dataset_augmented = None
        self.aggregated_data = None
        self.dataset = None
        self.GOAL_SMAE = 40
        self.training_features_input = None
        self.training_targets_input = None
        self.validation_features_input = None
        self.validation_targets_input = None
        self.scheduler = None
        self.target_transformer = None
        self.optimizer = None
        self.patience_limit = 150
        self.early_stopping_threshold = 1.0  # Set the threshold for early stopping
        self.csv_filename = "training_results.csv"
        self.val_dataset = None
        self.iteration_counter = 0
        self.train_dataset = None
        self.path_to_save_model = "."

        self.history_smae_val = []
        self.history_smae_train = []
        self.history_mae_val = []
        self.history_mse_val = []
        self.history_mse_train = []
        self.history_mae_train = []
        self.history_mae = []
        self.history_mae_train = []
        self.history_mse_train = []

        self.avg_val_mse = float('inf')
        self.avg_val_mae = float('inf')
        self.avg_val_rmse = float('inf')
        self.avg_val_smae = float('inf')
        self.avg_val_r2 = -1000
        self.best_val_mse = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_smae = float('inf')
        self.avg_train_mse = float('inf')
        self.avg_train_mae = float('inf')
        self.avg_train_rmse = float('inf')
        self.avg_train_smae = float('inf')
        self.best_train_mse = float('inf')
        self.best_train_mae = float('inf')
        self.best_train_smae = float('inf')

        self.target_scaler = None

        self.average_importances = None
        self.all_importances = None
        self.running_number = None
        self.validation_targets = None
        self.training_targets = None
        self.validation_data = None
        self.training_data = None
        self.giw_data = None
        self.average_estimated_depths = None
        self.name = "DepthEstimator"
        self.input_data = None
        self.model = None
        print("Device is: ", device)
        self.target_transformation = None
        self.transformers = None
        self.sequence_length = None

    def train_epoch(self, model, optimizer, train_loader, mse_loss_fn, mae_loss_fn, smae_loss_fn, epoch):
        # Training code for one epoch here
        model.train()
        total_samples = 0.0
        all_y_true = []
        all_y_pred = []
        total_mae = 0
        total_mse = 0
        total_smae = 0

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred, intermediate_activations = model(X_batch, return_intermediates=True)
            if torch.isnan(y_pred).any():
                raise ValueError('NaN values in model output')

            # Calculate loss on the scaled data
            smae_loss = smae_loss_fn(y_pred, y_batch)

            combined_loss = smae_loss
            combined_loss.backward()
            optimizer.step()

            # Inverse transform for metric calculation (post-backpropagation)
            y_pred_inv = self.inverse_transform_target(y_pred).to(device)
            y_batch_inv = self.inverse_transform_target(y_batch).to(device)

            # Accumulate metrics on the original scale
            total_mae += mae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_mse += mse_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_smae += smae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_samples += y_batch.size(0)

            all_y_true.append(y_batch_inv.detach().cpu().numpy())
            all_y_pred.append(y_pred_inv.detach().cpu().numpy())

        avg_train_mae = total_mae / total_samples
        avg_train_mse = total_mse / total_samples
        avg_train_smae = total_smae / total_samples
        avg_train_rmse = np.sqrt(avg_train_mse)
        avg_train_r2 = 1



        self.checkTrainResults(avg_train_mse, avg_train_mae, avg_train_smae, avg_train_rmse, avg_train_r2)
        return avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2

    def validate_epoch(self, model, val_loader, mse_loss_fn, mae_loss_fn, smae_loss_fn, patience_counter, epoch,
                       all_predictions, all_true_values):

        global all_true_values_array, all_predictions_array
        model.eval()

        total_val_mae = 0
        total_val_mse = 0
        total_val_smae = 0
        total_val_samples = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, intermediates = model(X_batch, return_intermediates=True)

                # y_pred = model(X_batch)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                # calculate loss
                val_mse = mse_loss_fn(y_pred, y_batch)
                val_mae = mae_loss_fn(y_pred, y_batch)
                val_smae = smae_loss_fn(y_pred, y_batch)

                # accumulate loss
                total_val_mae += val_mae.item() * y_batch.size(0)
                total_val_mse += val_mse.item() * y_batch.size(0)
                total_val_smae += val_smae.item() * y_batch.size(0)

                # count samples of batch
                total_val_samples += y_batch.size(0)

                # Assume errors is a list of absolute errors for each prediction
                errors = (y_pred - y_batch).cpu().numpy()

                # Log the biggest and smallest error
                max_error = np.max(errors)
                min_error = np.min(errors)
                median_error = np.median(errors)

                # Store predictions and true values
                all_predictions.append(y_pred.detach().cpu().numpy())
                all_true_values.append(y_batch.detach().cpu().numpy())



        try:
            all_predictions_array = np.concatenate(all_predictions)
            all_true_values_array = np.concatenate(all_true_values)
        except ValueError as e:
            # Log the error and the state of relevant variables
            print(f"Error: {e}")
            print(f"all_predictions length: {len(all_predictions)}")
            print(f"all_true_values length: {len(all_true_values)}")
            if all_predictions:
                print(f"Shape of first element in all_predictions: {all_predictions[0].shape}")
            if all_true_values:
                print(f"Shape of first element in all_true_values: {all_true_values[0].shape}")

        # Now calculate residuals
        residuals = np.abs(all_predictions_array - all_true_values_array)
        raw_residuals = all_predictions_array - all_true_values_array

        # Calculate R-squared for the entire dataset
        avg_val_r2 = r2_score(all_true_values_array, all_predictions_array)

        # EPOCH LOSSES
        avg_val_mae = total_val_mae / total_val_samples
        avg_val_mse = total_val_mse / total_val_samples
        avg_val_smae = total_val_smae / total_val_samples
        avg_val_rmse = np.sqrt(avg_val_mse)


        isBreakLoop, patience_counter = self.checkValidationResults(avg_val_mse, avg_val_mae, avg_val_smae, epoch,
                                                                    patience_counter, all_predictions_array,
                                                                    all_true_values_array)

        if (epoch >= n_epochs - 1) or isBreakLoop:
            # Save the activations
            save_activations_validation(intermediates, y_batch, "last", self.userFolder)
            analyzeResiduals(self.all_predictions_array, self.all_true_values_array)
            self.fold_results = None

        if isBreakLoop:
            print(f"Early stopping reached limit of {self.patience_limit}")
            return True, avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, patience_counter, residuals, raw_residuals

        return False, avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, patience_counter, residuals, raw_residuals

    def runFold(self, batch_size, embed_dim, learning_rate, weight_decay, l1_lambda, dropoutRate, fc1_dim, fold_count,
                n_splits, train_index, val_index, test_index, fold_performance, model=None, beta=0.5):

        print(f"Fold {fold_count}/{n_splits}")
        train_loader, valid_loader, test_loader, input_size = self.get_data_loader(train_index, val_index, test_index,
                                                                                   batch_size=batch_size)

        # Set target scaler for this object
        self.target_scaler = self.dataset_obj.target_scaler
        print("Target scaler is: ", self.target_scaler)
        print("Created data loaders")

        self.model, self.optimizer, self.scheduler = define_model_and_optim(model=model, input_size=input_size,
                                                                            embed_dim=embed_dim,
                                                                            dropoutRate=dropoutRate,
                                                                            learning_rate=learning_rate,
                                                                            weight_decay=weight_decay,
                                                                            fc1_dim=fc1_dim)
        print(self.model)
        print("Created model and optim")
        isContinueFold, goToNextOptimStep, self.best_val_smae = self.runConfiguredFold(batch_size, embed_dim,
                                                                                       dropoutRate, l1_lambda,
                                                                                       learning_rate, weight_decay,
                                                                                       fc1_dim, fold_performance,
                                                                                       train_loader, valid_loader,
                                                                                       test_loader,
                                                                                       beta=beta)

        return isContinueFold, goToNextOptimStep, self.best_val_smae

    def checkTrainResults(self, avg_train_mse, avg_train_mae, avg_train_smae, avg_train_rmse, avg_train_r2):

        # Check training results
        if float(avg_train_mae) < self.best_train_mae:
            self.best_train_mae = avg_train_mae

        if float(avg_train_smae) < self.best_train_smae:
            self.best_train_smae = avg_train_smae

        """ 3. Implement early stopping """
        if float(avg_train_mse) < self.best_train_mse:
            self.best_train_mse = avg_train_mse

    def checkValidationResults(self, avg_val_mse, avg_val_mae, avg_val_smae, epoch, patience_counter,
                               all_predictions_array, all_true_values_array):
        isBreakLoop = False
        # Check validation results
        if avg_val_mae < self.best_val_mae:
            self.best_val_mae = avg_val_mae

        if avg_val_smae < self.best_val_smae:
            self.best_val_smae = avg_val_smae
            torch.save(self.model.state_dict(), 'results/best_model_state_dict.pth')

            patience_counter = 0
            self.fold_results = analyzeResiduals(all_predictions_array, all_true_values_array)
            self.all_predictions_array = all_predictions_array
            self.all_true_values_array = all_true_values_array
            print(
                f'Model saved at epoch {epoch} with validation SMAE {self.best_val_smae:.6f} and MAE {self.best_val_mae}\n')
        else:
            patience_counter += 1

        """ 3. Implement early stopping """
        if avg_val_mse < self.best_val_mse:
            self.best_val_mse = avg_val_mse

        if avg_val_smae < self.early_stopping_threshold:
            isBreakLoop = True

        if patience_counter > self.patience_limit:
            isBreakLoop = True

        return isBreakLoop, patience_counter

    def inverse_transform_target(self, y_transformed):
        # Move the tensor to CPU if it's on GPU
        if y_transformed.is_cuda:
            y_transformed = y_transformed.cpu()

        # Now that the tensor is on the CPU, convert it to a NumPy array
        y_transformed_np = y_transformed.detach().numpy()

        # Reshape the array to a 2D array with a single column
        y_transformed_np_reshaped = y_transformed_np.reshape(-1, 1)

        if self.target_scaler is not None:
            # Perform the inverse transformation using the scaler
            y_inverse_transformed = self.target_scaler.inverse_transform(y_transformed_np_reshaped)

            # Flatten the array back to 1D
            y_inverse_transformed = y_inverse_transformed.flatten()

            # Convert the NumPy array back to a tensor
            return torch.from_numpy(y_inverse_transformed).to(device)
        else:
            return y_transformed

    def runConfiguredFold(self, batch_size, embed_dim, dropoutRate, l1_lambda, learning_rate, weight_decay, fc1_dim,
                          fold_performance, train_loader_0, valid_loader_0=None, test_loader_0=None, beta=0.5):
        goToNextOptimStep = False
        # Define loss functions
        mse_loss_fn = nn.MSELoss(reduction='sum').to(device)
        mae_loss_fn = nn.L1Loss().to(device)
        huber_loss_fn = nn.HuberLoss(delta=beta).to(device)
        smae_loss_fn = nn.SmoothL1Loss(beta=beta).to(device)

        isBreakLoop = False
        avg_train_r2 = -1000
        patience_counter = 0
        avg_train_mse = None
        avg_train_rmse = None
        avg_train_smae = None
        avg_train_mae = None

        avg_val_mse = None
        avg_val_rmse = None
        avg_val_mae = None
        avg_val_smae = None
        avg_val_r2 = None

        self.best_train_mse = float('inf')
        self.best_train_mae = float('inf')
        self.best_train_smae = float('inf')
        self.best_val_mse = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_smae = float('inf')

        for epoch in range(n_epochs):
            # Error analysis
            all_predictions = []
            all_true_values = []
            # train one epoch
            avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2 = self.train_epoch(
                self.model,
                self.optimizer,
                train_loader_0,
                mse_loss_fn,
                mae_loss_fn,
                smae_loss_fn,
                epoch)

            if keyboard.is_pressed('q'):
                goToNextOptimStep = True
                isBreakLoop = True
                # break  # Exit the outer loop to stop training completely

            if valid_loader_0 is not None:
                isBreakLoop, avg_val_mse, avg_val_rmse, avg_val_mae, avg_val_smae, avg_val_r2, patience_counter, residuals, raw_residuals = self.validate_epoch(
                    model=self.model, val_loader=valid_loader_0, mse_loss_fn=mse_loss_fn, mae_loss_fn=mae_loss_fn,
                    smae_loss_fn=smae_loss_fn, patience_counter=patience_counter, epoch=epoch,
                    all_predictions=all_predictions, all_true_values=all_true_values)

            if test_loader_0 is not None:
                print("Analysis of residuals for test data")
                all_true_values_array, all_predictions_array = self.test_new_data(model=self.model,
                                                                                  test_loader=test_loader_0,
                                                                                  mse_loss_fn=mse_loss_fn,
                                                                                  mae_loss_fn=mae_loss_fn,
                                                                                  smae_loss_fn=smae_loss_fn)
                # self.analyzeResiduals(all_predictions_array, all_true_values_array)
                print("\n\n")

            # Step through the scheduler at the end of each epoch
            self.scheduler.step()

            if isBreakLoop or epoch == n_epochs - 1:
                self.model.load_state_dict(torch.load('results/best_model_state_dict.pth'))
                model_path = os.path.join(self.userFolder, 'optimal_subject_model_state_dict.pth')
                torch.save(self.model.state_dict(), model_path)  # Saving state dictionary
                print("Optimal model state dictionary saved.")
                break

        # Results after all epochs
        print_results(self.iteration_counter, batch_size, embed_dim, dropoutRate, l1_lambda,
                      learning_rate,
                      weight_decay, fc1_dim, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae,
                      avg_train_r2,
                      self.best_train_mse, self.best_train_mae, self.best_train_smae, avg_val_mse,
                      avg_val_rmse,
                      avg_val_mae, avg_val_smae, avg_val_r2, self.best_val_mse, self.best_val_mae,
                      self.best_val_smae)

        # Store the metrics for this fold
        fold_performance.append({
            'fold': len(fold_performance) + 1,
            'avg_train_mse': avg_train_mse,
            'avg_train_rmse': avg_train_rmse,
            'avg_train_mae': avg_train_mae,
            'avg_train_smae': avg_train_smae,
            'avg_train_r2': avg_train_r2,
            'best_train_mse': self.best_train_mse,
            'best_train_mae': self.best_train_mae,
            'best_train_smae': self.best_train_smae,
            'avg_val_mse': avg_val_mse,
            'avg_val_rmse': avg_val_rmse,
            'avg_val_mae': avg_val_mae,
            'avg_val_smae': avg_val_smae,
            'avg_val_r2': avg_val_r2,
            'best_val_mse': self.best_val_mse,
            'best_val_mae': self.best_val_mae,
            'best_val_smae': self.best_val_smae
        })

        average_fold_val_smae = np.mean([f['best_val_smae'] for f in fold_performance])
        print(f"Average Validation SMAE across folds: {average_fold_val_smae}")

        average_fold_val_mae = np.mean([f['best_val_mae'] for f in fold_performance])
        print(f"Average Validation MAE across folds: {average_fold_val_mae}\n")

        if test_loader_0 is not None:
            all_true_values_array, all_predictions_array = self.test_new_data(model=self.model,
                                                                              test_loader=test_loader_0,
                                                                              mse_loss_fn=mse_loss_fn,
                                                                              mae_loss_fn=mae_loss_fn,
                                                                              smae_loss_fn=smae_loss_fn)

            with open('gt_values.pkl', 'wb') as file:
                pickle.dump(all_true_values_array, file)

            with open('pred_values.pkl', 'wb') as file:
                pickle.dump(all_predictions_array, file)

            analyzeResiduals(all_predictions_array, all_true_values_array)

        return True, goToNextOptimStep, self.best_val_smae

    def test_new_data(self, model, test_loader, mse_loss_fn, mae_loss_fn, smae_loss_fn):

        global all_predictions_array_test, all_true_values_array_test
        model.eval()
        all_predictions = []
        all_true_values = []

        total_test_mae = 0
        total_test_mse = 0
        total_test_smae = 0
        total_test_samples = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                # calculate loss
                test_mse = mse_loss_fn(y_pred, y_batch)
                test_mae = mae_loss_fn(y_pred, y_batch)
                test_smae = smae_loss_fn(y_pred, y_batch)

                # accumulate loss
                total_test_mae += test_mae.item() * y_batch.size(0)
                total_test_mse += test_mse.item() * y_batch.size(0)
                total_test_smae += test_smae.item() * y_batch.size(0)

                # count samples of batch
                total_test_samples += y_batch.size(0)

                # Assume errors is a list of absolute errors for each prediction
                errors = (y_pred - y_batch).cpu().numpy()

                # Log the biggest and smallest error
                max_error = np.max(errors)
                min_error = np.min(errors)
                median_error = np.median(errors)

                mode_result = stats.mode(errors, axis=None)
                mode_value = mode_result.mode[0]  # First element of the mode array
                mode_count = mode_result.count[0]  # First element of the count array

                all_predictions.append(y_pred.detach().cpu().numpy())
                all_true_values.append(y_batch.detach().cpu().numpy())


        try:
            all_predictions_array_test = np.concatenate(all_predictions)
            all_true_values_array_test = np.concatenate(all_true_values)
        except ValueError as e:
            # Log the error and the state of relevant variables
            print(f"Error: {e}")
            print(f"all_predictions length: {len(all_predictions)}")
            print(f"all_true_values length: {len(all_true_values)}")
            if all_predictions:
                print(f"Shape of first element in all_predictions: {all_predictions[0].shape}")
            if all_true_values:
                print(f"Shape of first element in all_true_values: {all_true_values[0].shape}")

            # Now calculate residuals
        residuals = np.abs(all_predictions_array_test - all_true_values_array_test)


        # Calculate R-squared for the entire dataset
        avg_test_r2 = r2_score(all_true_values_array_test, all_predictions_array_test)

        # EPOCH LOSSES
        avg_test_mae = total_test_mae / total_test_samples
        avg_test_mse = total_test_mse / total_test_samples
        avg_test_smae = total_test_smae / total_test_samples
        avg_test_rmse = np.sqrt(avg_test_mse)

        print(f"Average test MAE is {avg_test_mae}")

        # Check validation results
        if avg_test_mae < self.best_test_mae:
            self.best_test_mae = avg_test_mae

        if avg_test_mse < self.best_test_mse:
            self.best_test_mse = avg_test_mse

        if avg_test_smae < self.best_test_smae:
            self.best_test_smae = avg_test_smae
            # self.analyzeResiduals(all_predictions_array_test, all_true_values_array_test)

        if avg_test_mse < self.best_test_mse:
            self.best_val_mse = avg_test_mse

        return all_true_values_array_test, all_predictions_array_test

    def get_data_loader(self, train_index, val_index, test_index, batch_size=100):

        valid_loader_0 = None
        test_loader_0 = None

        train_subjects = train_index

        print("Training subjects: ", train_subjects)
        train_data = self.dataset[self.dataset['SubjectID'].isin(train_subjects)]
        train_data = create_features(train_data)
        train_data = subjective_normalization_dataset(train_data)
        train_data = self.dataset_obj.apply_transformation_dataset(train_data, isTrain=True)
        train_data = self.dataset_obj.scale_target_dataset(train_data, isTrain=True)
        assert len(train_data) > 0, "Training data is empty."
        train_sequences = create_sequences(train_data)
        assert len(train_sequences) > 0, "Training sequences are empty."
        train_features, train_targets = separate_features_and_targets(train_sequences)
        assert len(train_features) > 0 and len(train_targets) > 0, "Training features or targets are empty."

        # Example usage
        train_features_tensor, train_targets_tensor = create_lstm_tensors_dataset(train_features, train_targets)

        # get data loaders
        train_loader_0, input_size_0 = create_dataloaders_dataset(train_features_tensor, train_targets_tensor,
                                                                  batch_size=batch_size)
        self.train_loader = train_loader_0
        self.input_size = input_size_0

        if val_index is not None:
            # val_subjects = subject_list[val_index]
            val_subjects = val_index
            print("Valid subjects: ", val_subjects)
            val_subjects = [val_subjects] if isinstance(val_subjects, str) else val_subjects
            validation_data = self.dataset[self.dataset['SubjectID'].isin(val_subjects)]
            # Step 2.1: Create features!
            validation_data = create_features(validation_data)
            validation_data = subjective_normalization_dataset(validation_data)
            validation_data = self.dataset_obj.apply_transformation_dataset(validation_data, isTrain=False)
            validation_data = self.dataset_obj.scale_target_dataset(validation_data, isTrain=False)

            assert len(validation_data) > 0, "Validation data is empty."
            validation_sequences = create_sequences(validation_data)

            assert len(validation_sequences) > 0, "Validation sequences are empty."
            validation_features, validation_targets = separate_features_and_targets(
                validation_sequences)
            assert len(validation_features) > 0 and len(
                validation_targets) > 0, "Validation features or targets are empty."

            # Example usage
            valid_features_tensor, valid_targets_tensor = create_lstm_tensors_dataset(validation_features,
                                                                                      validation_targets)
            valid_loader_0, input_size_0 = create_dataloaders_dataset(valid_features_tensor, valid_targets_tensor,
                                                                      batch_size=batch_size)

            self.valid_loader = valid_loader_0

        if test_index is not None:
            # val_subjects = subject_list[val_index]
            test_subjects = test_index
            print("Test subjects: ", test_subjects)
            test_subjects = [test_subjects] if isinstance(test_subjects, str) else test_subjects
            test_data = self.dataset[self.dataset['SubjectID'].isin(test_subjects)]
            # Step 2.1: Create features!
            test_data = create_features(test_data)
            test_data = subjective_normalization_dataset(test_data)
            test_data = self.dataset_obj.apply_transformation_dataset(test_data, isTrain=False)
            test_data = self.dataset_obj.scale_target_dataset(test_data, isTrain=False)

            assert len(test_data) > 0, "Test data is empty."
            test_sequences = create_sequences(test_data, sequence_length=10)

            assert len(test_sequences) > 0, "Test sequences are empty."
            test_features, test_targets = separate_features_and_targets(test_sequences)
            assert len(test_features) > 0 and len(
                test_targets) > 0, "Validation features or targets are empty."

            # Example usage
            test_features_tensor, test_targets_tensor = create_lstm_tensors_dataset(test_features,
                                                                                    test_targets)
            test_loader_0, input_size_0 = create_dataloaders_dataset(test_features_tensor, test_targets_tensor,
                                                                     batch_size=batch_size)

        return train_loader_0, valid_loader_0, test_loader_0, input_size_0
