import datetime
import json
import pickle
import os
import csv

import torch
import numpy as np
from torch import nn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from data.AbstractDatasetClass import AbstractDatasetClass

# from data.AbstractDatasetClass import AbstractDatasetClass
from data.robustVision_dataset import RobustVisionDataset
from data.utilities import create_optimizer, analyzeResiduals
from models.foval import Foval
from torch.cuda.amp import autocast, GradScaler


class FOVALTrainer:
    def __init__(self, config_path, dataset: AbstractDatasetClass, device, feature_names,
                 save_intermediates_every_epoch, model_type):
        """
        Initialize the FOVALTrainer with feature count, dataset object, and model save path.
        """
        self.per_fold_results = []
        self.current_fold = None
        self.model_type = model_type
        self.hidden_layer_size = None
        self.dropout_rate = None
        self.fc1_dim = None
        self.patience_counter = 0
        self.early_stopping_threshold = 1.0
        self.patience_limit = 150  # originally 150
        self.hyperparameters = None
        self.feature_count = len(feature_names) - 2  # as we need to remove target and subject column
        self.feature_names = feature_names
        self.dataset = dataset  # Dataset object from RobustVisionDataset
        self.config_path = config_path
        self.save_path = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.fold_results = []
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}

        self.current_metrics = {}

        self.target_scaler = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.sequence_length = 10 #self.dataset.sequence_length
        self.csv_filename = "training_results.csv"
        self.device = device
        print(f"Device is: {self.device}")

        if self.dataset.subject_list is not None:
            self.n_splits = len(self.dataset.subject_list)
        else:
            self.n_splits = 5

        print("Number of splits: ", self.n_splits)

        self.beta = None
        self.weight_decay = None
        self.learning_rate = None
        self.batch_size = None
        self.l1_lambda = None

        # Flags
        self.save_intermediates_every_epoch = save_intermediates_every_epoch

    def setup(self):
        self.load_model_checkpoint(self.config_path)
        self.initialize_model()

    def load_model_checkpoint(self, config_path):
        with open(config_path, 'r') as f:
            hyper_parameters = json.load(f)
        self.hyperparameters = hyper_parameters
        self.batch_size =  hyper_parameters['batch_size']
        self.learning_rate = hyper_parameters['learning_rate']
        self.weight_decay = hyper_parameters['weight_decay']
        self.fc1_dim = hyper_parameters['fc1_dim']
        self.dropout_rate = hyper_parameters['dropout_rate']
        self.hidden_layer_size = hyper_parameters['embed_dim']
        self.beta = 0.75

        print("Hyper parameters: ", hyper_parameters)

    def initialize_model(self):
        self.model = Foval(device=self.device, feature_count=self.feature_count, model_type=self.model_type)
        self.model.initialize(
            input_size=self.feature_count,
            fc1_dim=self.hyperparameters['fc1_dim'],
            dropout_rate=self.hyperparameters['dropout_rate'],
            hidden_layer_size=self.hyperparameters['embed_dim']
        )

    def save_activations_and_weights(self, intermediates, filename, file_path):
        """

        @param intermediates:
        @param filename:
        @param file_path:
        """
        save_path_tensors = os.path.join(file_path, f"{filename}_activations.pt")
        save_path_numpy = os.path.join(file_path, f"{filename}_weights.pkl")

        # Separate tensors and numpy arrays
        tensors_dict = {k: v for k, v in intermediates.items() if isinstance(v, torch.Tensor)}
        numpy_dict = {k: v for k, v in intermediates.items() if isinstance(v, np.ndarray)}

        # Save tensors using torch.save
        torch.save(tensors_dict, save_path_tensors)

        # Save numpy arrays using pickle
        with open(save_path_numpy, 'wb') as f:
            pickle.dump(numpy_dict, f)

        print(f"Tensors saved to {save_path_tensors}")
        print(f"NumPy arrays saved to {save_path_numpy}")

    def cross_validate_with_specific_datasets(self, num_epochs=10):
        """
        Perform cross-validation with predefined training and validation datasets.
        """
        train_subjects = self.dataset.train_data['SubjectID'].unique()
        val_subjects = self.dataset.val_data['SubjectID'].unique()

        fold_accuracies = []

        print("Starting cross-validation with separate datasets.")

        for fold in range(len(val_subjects)):
            print(f"\n\nStarting Fold {fold + 1}/{len(val_subjects)}")

            train_subject_ids = list(train_subjects)
            val_subject_id = [val_subjects[fold]]

            self.current_fold = fold + 1

            fold_mae = self.run_fold(train_subject_ids, val_subject_id, None, num_epochs)
            fold_accuracies.append(fold_mae)
            print(f"Fold {fold + 1} MAE: {fold_mae}")
            average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
            print(f"Average Validation MAE across folds: {average_accuracy}")
            self.reset_metrics()

        best_fold = min(fold_accuracies)
        print(f"Best Fold with MAE: {best_fold}")
        average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print(f"Average Cross-Validation MSE: {average_accuracy}")
        return average_accuracy
    

    # BECAUSE OF DIFFERENT RANGES IN THE DATASET, WE NEED TO USE THE SAME RANGE FOR ALL THE DATASETS
    # BY USING THE TRAINING RANGE AS THE VALIDATION RANGE
    def restrict_validation_to_training_range(train_data, val_data, target_column='Gt_Depth'):
        """
        Restricts the validation dataset to the range of the target variable in the training dataset.
        """
        train_min = train_data[target_column].min()
        train_max = train_data[target_column].max()

        # Filter validation data
        restricted_val_data = val_data[(val_data[target_column] >= train_min) & (val_data[target_column] <= train_max)]

        print(f"Validation data restricted to range: {train_min:.2f} - {train_max:.2f}")
        print(f"Restricted validation dataset size: {len(restricted_val_data)} rows")

        return restricted_val_data

    def cross_validate(self, num_epochs=10, start_fold=0):
        """
        Perform cross-validation on the dataset, aber erst ab `start_fold`.
        """
        # Erzeuge alle Splits einmal
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        all_splits = list(kf.split(self.dataset.subject_list))

        fold_accuracies = []

        # Durchlaufe nur die Splits ab `start_fold`
        for fold_idx in range(start_fold, len(all_splits)):
            train_idx, val_idx = all_splits[fold_idx]
            print(f"\n\nStarting Fold {fold_idx + 1}/{self.n_splits}")

            train_subjects = list(self.dataset.subject_list[train_idx])
            val_subjects   = list(self.dataset.subject_list[val_idx])

            print("Train Subjects: ", train_subjects)
            print("Validation Subjects: ", val_subjects)

            # Set current fold
            self.current_fold = fold_idx + 1

            # Training/Validierung ausführen
            fold_mae = self.run_fold(train_subjects, val_subjects, None, num_epochs)
            fold_accuracies.append(fold_mae)

            print(f"Fold {fold_idx + 1} MAE: {fold_mae}")
            avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
            print(f"Average Validation MAE across folds: {avg_accuracy}")

            # Resette ggf. Metriken
            self.reset_metrics()

        # Am Ende noch die Gesamtauswertung
        best_fold = min(fold_accuracies)
        print(f"Best Fold with MAE: {best_fold}")
        overall_avg = sum(fold_accuracies) / len(fold_accuracies)
        print(f"Average Cross-Validation MAE: {overall_avg}")

        self.write_results_to_csv()
        return overall_avg

    def write_results_to_csv(self, filename="summary_model_results.csv"):
        result_path = os.path.join("results", filename)
        keys = ["Model", "Dataset", "ValidationSubject", "MAE", "RMSE"]
        write_header = not os.path.exists(result_path) or os.stat(result_path).st_size == 0

        with open(result_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerows(self.per_fold_results)

    def run_fold(self, train_index, val_index=None, test_index=None, num_epochs=10):
        # Ensure that val_index is not None and has elements
        if val_index is not None and len(val_index) > 0:
            validation_participant_name = val_index[0]
            training_participant_name = train_index

            print("Training Participants Names: ", training_participant_name)
            print("Validation Participant Name: ", validation_participant_name, "\n")

        else:
            validation_participant_name = "unknown"

        # Set the save path using the validation participant's name
        self.save_path = os.path.join("results", validation_participant_name)
        os.makedirs(self.save_path, exist_ok=True)  # Create the directory if it doesn't exist
        print("Save path is set to: ", self.save_path)

        print(f"Train index: {train_index} and val index {validation_participant_name}, and test index {test_index}, and batch size {self.batch_size}")
        # Prepare data loaders
        self.train_loader, self.valid_loader, input_size = self.dataset.get_data_loader(
            train_index, validation_participant_name, self.batch_size)

        self.target_scaler = self.dataset.target_scaler
        self.optimizer, self.scheduler = create_optimizer(
            model=self.model, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay)

        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as profiler:
            for epoch in range(num_epochs):
                self.train_epoch(epoch)

                if self.valid_loader:
                    is_last_epoch = (epoch == num_epochs - 1)
                    self.validate_epoch(epoch, val_subject=validation_participant_name, is_last_epoch=is_last_epoch)

                self.scheduler.step()

                if self.check_early_stopping(epoch):
                    break
            profiler.step()  # Step profiler at the end of each epoch

        # Save model state dictionary after training
        self.save_model_state(epoch)
        # print validation SMAE and MAE averages of epoch
        # average_fold_val_smae = np.mean([f['best_val_smae'] for f in fold_performance])
        # print(f"Average Validation SMAE across folds: {average_fold_val_smae}")
        #
        # average_fold_val_mae = np.mean([f['best_val_mae'] for f in fold_performance])
        # print(f"Average Validation MAE across folds: {average_fold_val_mae}\n")

        return self.best_metrics["mae"]

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        """
        scaler = GradScaler()  # Initialize GradScaler for mixed precision

        mse_loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        mae_loss_fn = nn.L1Loss().to(self.device)
        smae_loss_fn = nn.SmoothL1Loss(beta=0.75).to(self.device)

        self.model.train()
        total_samples = 0.0
        total_mae, total_mse, total_smae = 0, 0, 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()

            # Example forward pass with mixed precision
            with autocast():
                y_pred, _ = self.model(X_batch, return_intermediates=True)
                smae_loss = smae_loss_fn(y_pred, y_batch)
            # Scaled backward pass
            scaler.scale(smae_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # smae_loss.backward()
            # self.optimizer.step()

            # Inverse transform for metric calculation (post-backpropagation)
            y_pred_inv = self.inverse_transform_target(y_pred)
            y_batch_inv = self.inverse_transform_target(y_batch)

            # Accumulate metrics on the original scale
            total_mae += mae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_mse += mse_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_smae += smae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_samples += y_batch.size(0)

        self.current_metrics["train_mae"] = total_mae / total_samples
        self.current_metrics["train_mse"] = total_mse / total_samples
        self.current_metrics["train_smae"] = total_smae / total_samples

    def validate_epoch(self, epoch, val_subject, is_last_epoch=False):
        mse_loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        mae_loss_fn = nn.L1Loss().to(self.device)
        smae_loss_fn = nn.SmoothL1Loss().to(self.device)

        self.model.eval()
        total_val_mae, total_val_mse, total_val_smae = 0, 0, 0
        total_val_samples = 0.0
        all_predictions, all_true_values = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.valid_loader:

                # if keyboard.is_pressed('q'):
                #    break
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred, intermediates = self.model(X_batch, return_intermediates=True)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                total_val_mae += mae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_mse += mse_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_smae += smae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_samples += y_batch.size(0)

                all_predictions.append(y_pred.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

        # Store metrics for this epoch
        self.current_metrics["val_mae"] = total_val_mae / total_val_samples
        self.current_metrics["val_mse"] = total_val_mse / total_val_samples
        self.current_metrics["val_smae"] = total_val_smae / total_val_samples
        true_vals = np.concatenate(all_true_values)
        pred_vals = np.concatenate(all_predictions)
        mask = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
        self.current_metrics["val_r2"] = r2_score(true_vals[mask], pred_vals[mask])

        # rmse = np.sqrt(mean_squared_error(true_vals[mask], pred_vals[mask]))


        print("Saving fold results of ", val_subject)
        self.log_epoch_metrics(epoch, val_subject, int(total_val_samples))

        # Save activations and weights based on the condition
        # if self.save_intermediates_every_epoch or is_last_epoch:
        #     self.save_activations_and_weights(intermediates, "intermediates", self.save_path)

    def log_epoch_metrics(self, epoch, val_subj, num_samples):
        """Log metrics for each epoch and subject into a CSV file."""
        log_path = os.path.join("results", "epoch_logs.csv")
        keys = [
            "Timestamp", "Model", "Dataset", "ValidationSubject", "Fold", "Epoch",
            "MAE", "MSE", "SMAE", "RMSE", "R2",
            "Train_MAE", "Train_MSE", "Train_SMAE",
            "Samples"
        ]

        # Ensure directory exists
        os.makedirs("results", exist_ok=True)

        # Calculate RMSE if true values exist
        val_mae = self.current_metrics.get("val_mae", None)
        val_mse = self.current_metrics.get("val_mse", None)
        val_smae = self.current_metrics.get("val_smae", None)
        val_r2 = self.current_metrics.get("val_r2", None)
        rmse = np.sqrt(val_mse) if val_mse is not None else None

        row = {
            "Timestamp": datetime.datetime.now().isoformat(),
            "Model": self.model_type,
            "Dataset": getattr(self.dataset, "dataset_name", "Unknown"),
            "ValidationSubject": val_subj,
            "Fold": self.current_fold,
            "Epoch": epoch,
            "MAE": round(val_mae, 4) if val_mae else None,
            "MSE": round(val_mse, 4) if val_mse else None,
            "SMAE": round(val_smae, 4) if val_smae else None,
            "RMSE": round(rmse, 4) if rmse else None,
            "R2": round(val_r2, 4) if val_r2 else None,
            "Train_MAE": round(self.current_metrics.get("train_mae", 0), 4),
            "Train_MSE": round(self.current_metrics.get("train_mse", 0), 4),
            "Train_SMAE": round(self.current_metrics.get("train_smae", 0), 4),
            "Samples": num_samples
        }

        write_header = not os.path.exists(log_path)
        with open(log_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def inverse_transform_target(self, y_transformed):
        """
        Apply inverse transformation to the target variable.
        """
        if y_transformed.is_cuda:
            y_transformed = y_transformed.cpu()

        # y_transformed_np = y_transformed.detach().numpy().reshape(-1, 1)
        y_transformed_np = y_transformed.detach().cpu().numpy().reshape(-1, 1)

        if self.dataset.target_scaler:
            y_inverse_transformed = self.dataset.target_scaler.inverse_transform(y_transformed_np).flatten()
            return torch.from_numpy(y_inverse_transformed).to(self.device)
        return y_transformed

    def check_early_stopping(self, epoch):
        isBreakLoop = False

        # Check if the current SMAE is better than the best one
        if self.current_metrics["val_mae"] < self.best_metrics["mae"]:
            # Update best SMAE and save model
            self.best_metrics["smae"] = self.current_metrics["val_smae"]
            self.best_metrics["mae"] = self.current_metrics["val_mae"]

            # torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model_state_dict.pth'))
            print(
                f"Model saved at epoch {epoch} with SMAE {self.current_metrics['val_smae']} and MAE {self.current_metrics['val_mae']}")
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            print(f"Current Epoch: {epoch}")

        # Early stopping logic
        if self.current_metrics[
            "val_mae"] < self.early_stopping_threshold or self.patience_counter > self.patience_limit:
            isBreakLoop = True


        return isBreakLoop
        #
        # isBreakLoop = False
        # # Check validation results
        # if self.current_metrics["val_mae"] < self.best_metrics["mae"]:
        #     self.best_metrics["mae"] = self.current_metrics["val_mae"]
        #
        # if self.current_metrics["val_smae"] < self.best_metrics["smae"]:
        #     self.best_metrics["smae"] = self.current_metrics["val_smae"]
        #
        #     torch.save(self.model.state_dict(), 'results/best_model_state_dict.pth')
        #
        #     self.patience_counter = 0
        #     # self.fold_results = analyzeResiduals(all_predictions_array, all_true_values_array)
        #     # self.all_predictions_array = all_predictions_array
        #     # self.all_true_values_array = all_true_values_array
        #     print(
        #         f'Model saved at epoch {epoch} with validation SMAE {self.best_metrics["smae"]:.6f} and MAE {self.best_metrics["mae"]}\n')
        # else:
        #     self.patience_counter += 1
        #
        # """ 3. Implement early stopping """
        # if self.current_metrics["val_mse"] < self.best_metrics["mae"]:
        #     self.best_metrics["mae"] = self.current_metrics["val_mse"]
        #
        # if self.current_metrics["val_smae"] < self.early_stopping_threshold:
        #     isBreakLoop = True
        #
        # if self.patience_counter > self.patience_limit:
        #     isBreakLoop = True

        return isBreakLoop, self.patience_counter

    def analyzeResiduals(predictions, actual_values):
        # Calculate absolute errors
        absolute_errors = np.abs(predictions - actual_values)

        # 1. CALCULATE THE AMOUNT OF GREAT OK AND BAD errors
        """
            Desc: Show how many good, ok, and bad estimations we have (remember to show distribution too) 
            Categorize errors into bins
            bin 1: < 1 cm
            bin 2: < 10 cm
            bin 3: > 20 cm
        """
        bin1 = 1
        bin2 = 10
        bin3 = 20

        errors_under_1cm = np.sum(absolute_errors < bin1) / len(absolute_errors)
        errors_1cm_to_10cm = np.sum((absolute_errors >= bin1) & (absolute_errors <= bin2)) / len(absolute_errors)
        errors_10cm_to_20cm = np.sum((absolute_errors >= bin2) & (absolute_errors < bin3)) / len(absolute_errors)
        errors_above_20cm = np.sum(absolute_errors > bin3) / len(absolute_errors)

        # Print percentages
        print(f"Errors under 1 cm: {errors_under_1cm * 100:.2f}%")
        print(f"Errors between 1 cm and 10 cm: {errors_1cm_to_10cm * 100:.2f}%")
        print(f"Errors between 10 cm and 20 cm: {errors_10cm_to_20cm * 100:.2f}%")
        print(f"Errors above 20 cm: {errors_above_20cm * 100:.2f}%")

        # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
        # 2. Calculate which ranges (in bins of 10 cm )where predicted with wich average error
        """
            Desc: Show which depth values can be predicted the best
            Analyze performance across different depths
            bin size: 10
        """
        bin_size = 10
        # Bin actual values into 10 cm intervals
        bins = np.arange(30, max(actual_values) + bin_size, bin_size)  # Adjust the range as needed
        bin_indices = np.digitize(actual_values, bins)

        # Calculate mean absolute error for each bin
        mean_errors_per_bin = []

        for i in range(1, len(bins)):
            bin_errors = absolute_errors[bin_indices == i]
            mean_error = np.nanmean(bin_errors) if len(bin_errors) > 0 else 0.0
            if mean_error != 0.0:
                mean_errors_per_bin.append(mean_error)

        # Print mean errors per bin
        for i, error in enumerate(mean_errors_per_bin):
            print(f"Depths bin: {bins[i]} to {bins[i + 1]} cm: MAE: {error:.2f} cm")

        # calculate average error between 0.35 and 2 meters
        lower_bound = 35
        upper_bound = 200

        # Filter absolute_errors based on the condition that actual_values are between 0.35m and 2m
        filtered_errors_for_specific_actual_range = absolute_errors[
            (actual_values >= lower_bound) & (actual_values <= upper_bound)]

        # Calculate the mean of these filtered errors
        average_error_for_specific_actual_range = np.mean(filtered_errors_for_specific_actual_range)

        print(
            f"MAE for depth range between {lower_bound} and {upper_bound} cm: {average_error_for_specific_actual_range:.2f} cm")

        # calculate average error between 0 and 6 meters
        lower_bound = 0
        upper_bound = 600

        # Filter absolute_errors based on the condition that actual_values are between 0.35m and 2m
        filtered_errors_for_specific_actual_range = absolute_errors[
            (actual_values >= lower_bound) & (actual_values <= upper_bound)]

        # Calculate the mean of these filtered errors
        average_error_for_specific_actual_range = np.mean(filtered_errors_for_specific_actual_range)

        print(
            f"MAE for depth range between {lower_bound} and {upper_bound} cm: {average_error_for_specific_actual_range:.2f} cm")

        results = {
            '<1cm': errors_under_1cm * 100,
            '1-10cm': errors_1cm_to_10cm * 100,
            '10-20': errors_10cm_to_20cm * 100,
            '>20': errors_above_20cm * 100,
            'mean_errors_per_bin': {f"{bins[i]} to {bins[i + 1]} cm": error for i, error in
                                    enumerate(mean_errors_per_bin)},
            'average_error_for_2m_range': average_error_for_specific_actual_range
        }
        return results

    def save_model_state(self, epoch):
        """
        Save the model state dictionary after training.
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model_state_dict.pth')))
        model_path = os.path.join(self.save_path, 'optimal_subject_model_state_dict.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Optimal model state dictionary saved at epoch {epoch}.")

    def set_save_path(self, fold_name):
        """
        Set the save path for the current fold.
        """
        # Create a directory for the current fold under results/
        self.save_path = os.path.join("results", fold_name)

        # Ensure the directory exists
        os.makedirs(self.save_path, exist_ok=True)

        print(f"Save path set to: {self.save_path}")

    def reset_metrics(self):
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}
        # self.current_metrics = {"val_smae": float('inf'), "mse": float('inf'), "val_mae": float('inf')}
