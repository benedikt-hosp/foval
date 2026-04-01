import datetime
import json
import os
import csv

import torch
import numpy as np
from torch import nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from data.AbstractDatasetClass import AbstractDatasetClass
from data.utilities import create_optimizer
from models.foval import Foval
from torch.cuda.amp import autocast, GradScaler


class FOVALTrainer:
    def __init__(self, config_path, dataset: AbstractDatasetClass, device, feature_names,
                 save_intermediates_every_epoch, model_type):
        self.per_fold_results = []
        self.current_fold = None
        self.model_type = model_type
        self.hidden_layer_size = None
        self.dropout_rate = None
        self.fc1_dim = None
        self.patience_counter = 0
        self.early_stopping_threshold = 1.0
        self.patience_limit = 150
        self.hyperparameters = None
        self.feature_count = len(feature_names) - 2  # remove target and subject column
        self.feature_names = feature_names
        self.dataset = dataset
        self.config_path = config_path
        self.save_path = None

        self.train_loader = None
        self.valid_loader = None

        self.fold_results = []
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}
        self.current_metrics = {}

        self.target_scaler = None
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.sequence_length = 10
        self.device = device
        print(f"Device is: {self.device}")

        self.n_splits = len(self.dataset.subject_list) if self.dataset.subject_list is not None else 5
        print("Number of splits: ", self.n_splits)

        self.beta = None
        self.weight_decay = None
        self.learning_rate = None
        self.batch_size = None
        self.l1_lambda = None

        self.save_intermediates_every_epoch = save_intermediates_every_epoch

    def setup(self):
        self.load_model_checkpoint(self.config_path)
        self.initialize_model()

    def load_model_checkpoint(self, config_path):
        with open(config_path, 'r') as f:
            hyper_parameters = json.load(f)
        self.hyperparameters = hyper_parameters
        self.batch_size = hyper_parameters['batch_size']
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

    def cross_validate_with_specific_datasets(self, num_epochs=10):
        train_subjects = self.dataset.train_data['SubjectID'].unique()
        val_subjects = self.dataset.val_data['SubjectID'].unique()
        fold_accuracies = []

        print("Starting cross-validation with separate datasets.")

        for fold in range(len(val_subjects)):
            print(f"\n\nStarting Fold {fold + 1}/{len(val_subjects)}")
            self.current_fold = fold + 1

            fold_mae = self.run_fold(list(train_subjects), [val_subjects[fold]], None, num_epochs)
            fold_accuracies.append(fold_mae)

            print(f"Fold {fold + 1} MAE: {fold_mae}")
            print(f"Average Validation MAE across folds: {sum(fold_accuracies) / len(fold_accuracies)}")
            self.reset_metrics()

        best_fold = min(fold_accuracies)
        average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print(f"Best Fold with MAE: {best_fold}")
        print(f"Average Cross-Validation MAE: {average_accuracy}")
        return average_accuracy

    def cross_validate(self, num_epochs=10, start_fold=0):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        all_splits = list(kf.split(self.dataset.subject_list))
        fold_accuracies = []

        for fold_idx in range(start_fold, len(all_splits)):
            train_idx, val_idx = all_splits[fold_idx]
            print(f"\n\nStarting Fold {fold_idx + 1}/{self.n_splits}")

            train_subjects = list(self.dataset.subject_list[train_idx])
            val_subjects = list(self.dataset.subject_list[val_idx])

            print("Train Subjects: ", train_subjects)
            print("Validation Subjects: ", val_subjects)

            self.current_fold = fold_idx + 1
            fold_mae = self.run_fold(train_subjects, val_subjects, None, num_epochs)
            fold_accuracies.append(fold_mae)

            print(f"Fold {fold_idx + 1} MAE: {fold_mae}")
            print(f"Average Validation MAE across folds: {sum(fold_accuracies) / len(fold_accuracies)}")
            self.reset_metrics()

        best_fold = min(fold_accuracies)
        overall_avg = sum(fold_accuracies) / len(fold_accuracies)
        print(f"Best Fold with MAE: {best_fold}")
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
        validation_participant_name = val_index[0] if val_index and len(val_index) > 0 else "unknown"

        print("Training Participants Names: ", train_index)
        print("Validation Participant Name: ", validation_participant_name, "\n")

        self.save_path = os.path.join("results", validation_participant_name)
        os.makedirs(self.save_path, exist_ok=True)
        print("Save path is set to: ", self.save_path)

        print(f"Train index: {train_index} and val index {validation_participant_name}, "
              f"and test index {test_index}, and batch size {self.batch_size}")

        self.train_loader, self.valid_loader, input_size = self.dataset.get_data_loader(
            train_index, validation_participant_name, self.batch_size)

        self.target_scaler = self.dataset.target_scaler
        self.optimizer, self.scheduler = create_optimizer(
            model=self.model, learning_rate=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(num_epochs):
            self.train_epoch(epoch)

            if self.valid_loader:
                is_last_epoch = (epoch == num_epochs - 1)
                self.validate_epoch(epoch, val_subject=validation_participant_name, is_last_epoch=is_last_epoch)

            self.scheduler.step()

            if self.check_early_stopping(epoch):
                break

        self.save_model_state(epoch)
        return self.best_metrics["mae"]

    def train_epoch(self, epoch):
        scaler = GradScaler()
        smae_loss_fn = nn.SmoothL1Loss(beta=0.75).to(self.device)
        mae_loss_fn = nn.L1Loss().to(self.device)
        mse_loss_fn = nn.MSELoss(reduction='sum').to(self.device)

        self.model.train()
        total_samples = 0.0
        total_mae, total_mse, total_smae = 0, 0, 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                y_pred, _ = self.model(X_batch, return_intermediates=True)
                smae_loss = smae_loss_fn(y_pred, y_batch)

            scaler.scale(smae_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            y_pred_inv = self.inverse_transform_target(y_pred)
            y_batch_inv = self.inverse_transform_target(y_batch)

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
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred, _ = self.model(X_batch, return_intermediates=True)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                total_val_mae += mae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_mse += mse_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_smae += smae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_samples += y_batch.size(0)

                all_predictions.append(y_pred.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

        self.current_metrics["val_mae"] = total_val_mae / total_val_samples
        self.current_metrics["val_mse"] = total_val_mse / total_val_samples
        self.current_metrics["val_smae"] = total_val_smae / total_val_samples

        true_vals = np.concatenate(all_true_values)
        pred_vals = np.concatenate(all_predictions)
        mask = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
        self.current_metrics["val_r2"] = r2_score(true_vals[mask], pred_vals[mask])

        self.log_epoch_metrics(epoch, val_subject, int(total_val_samples))

    def log_epoch_metrics(self, epoch, val_subj, num_samples):
        log_path = os.path.join("results", "epoch_logs.csv")
        keys = [
            "Timestamp", "Model", "Dataset", "ValidationSubject", "Fold", "Epoch",
            "MAE", "MSE", "SMAE", "RMSE", "R2",
            "Train_MAE", "Train_MSE", "Train_SMAE", "Samples"
        ]

        os.makedirs("results", exist_ok=True)

        val_mae = self.current_metrics.get("val_mae")
        val_mse = self.current_metrics.get("val_mse")
        val_smae = self.current_metrics.get("val_smae")
        val_r2 = self.current_metrics.get("val_r2")
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
        if y_transformed.is_cuda:
            y_transformed = y_transformed.cpu()

        y_np = y_transformed.detach().cpu().numpy().reshape(-1, 1)

        if self.dataset.target_scaler:
            y_inverse = self.dataset.target_scaler.inverse_transform(y_np).flatten()
            return torch.from_numpy(y_inverse).to(self.device)
        return y_transformed

    def check_early_stopping(self, epoch):
        val_mae = self.current_metrics["val_mae"]
        val_smae = self.current_metrics["val_smae"]
        train_mae = self.current_metrics.get("train_mae", 0)

        if val_mae < self.best_metrics["mae"]:
            self.best_metrics["smae"] = val_smae
            self.best_metrics["mae"] = val_mae
            print(f"  Epoch {epoch:4d} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | * new best")
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if epoch % 25 == 0:
                print(f"  Epoch {epoch:4d} | Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | patience: {self.patience_counter}/{self.patience_limit}")

        if val_mae < self.early_stopping_threshold:
            print(f"  Early stop: MAE {val_mae:.2f} < threshold {self.early_stopping_threshold}")
            return True
        if self.patience_counter > self.patience_limit:
            print(f"  Early stop: patience exhausted at epoch {epoch}")
            return True

        return False

    def save_model_state(self, epoch):
        model_path = os.path.join(self.save_path, 'optimal_subject_model_state_dict.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Optimal model state dictionary saved at epoch {epoch}.")

    def reset_metrics(self):
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}
