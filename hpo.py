"""
Hyperparameter Optimization with Optuna.

Uses PreprocessedDataset (all 3 datasets combined) with 5-fold CV.
Results are stored in a SQLite DB so trials can be resumed later.

Usage:
    python hpo.py                    # run 20 trials
    python hpo.py --n_trials 50      # run 50 trials
    python hpo.py --resume           # continue from previous study
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import torch
from sklearn.model_selection import KFold

from main import seed_everything, setup_device
from data.PreprocessedDataset import PreprocessedDataset
from data.foval_preprocessor import separate_features_and_targets
from data.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset
from models.foval import Foval
from torch.cuda.amp import autocast, GradScaler

DB_PATH = "sqlite:///hpo_results.db"
STUDY_NAME = "foval_hpo"
N_FOLDS = 5
N_EPOCHS = 100
PATIENCE = 30
EARLY_STOP_THRESHOLD = 1.0


def create_model_and_optimizer(trial, device, feature_count):
    lr = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
    embed_dim = trial.suggest_int("embed_dim", 128, 2048, step=64)
    fc1_dim = trial.suggest_int("fc1_dim", 128, 2048, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.2, log=True)

    model = Foval(device=device, feature_count=feature_count, model_type="LSTM")
    model.initialize(
        input_size=feature_count,
        fc1_dim=fc1_dim,
        dropout_rate=dropout_rate,
        hidden_layer_size=embed_dim,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    return model, optimizer, scheduler, batch_size


def train_one_fold(model, optimizer, scheduler, dataset, train_subjects, val_subjects,
                   batch_size, device, trial, fold_idx):
    """Train one fold, return best val MAE."""
    train_loader = dataset.prepare_loader(train_subjects, batch_size, is_train=True)
    val_loader = dataset.prepare_loader(val_subjects, batch_size, is_train=False)

    smae_loss_fn = torch.nn.SmoothL1Loss(beta=0.75).to(device)
    mae_loss_fn = torch.nn.L1Loss().to(device)
    scaler = GradScaler()

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(N_EPOCHS):
        # --- Train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with autocast():
                y_pred, _ = model(X_batch, return_intermediates=True)
                loss = smae_loss_fn(y_pred, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # --- Validate ---
        model.eval()
        total_mae, total_samples = 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred, _ = model(X_batch, return_intermediates=True)

                # Inverse transform for real-scale MAE
                y_pred_inv = inverse_transform(y_pred, dataset, device)
                y_batch_inv = inverse_transform(y_batch, dataset, device)

                total_mae += mae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
                total_samples += y_batch.size(0)

        val_mae = total_mae / total_samples

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1

        # Pruning: report intermediate value
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_mae < EARLY_STOP_THRESHOLD:
            break
        if patience_counter >= PATIENCE:
            break

    return best_val_mae


def inverse_transform(y, dataset, device):
    if y.is_cuda:
        y = y.cpu()
    y_np = y.detach().cpu().numpy().reshape(-1, 1)
    if dataset.target_scaler:
        y_inv = dataset.target_scaler.inverse_transform(y_np).flatten()
        return torch.from_numpy(y_inv).to(device)
    return y.to(device)


def objective(trial):
    seed_everything(42)
    device = setup_device()

    dataset = PreprocessedDataset(
        parquet_dir="data/preprocessed",
        datasets=["giw", "robustvision", "tufts"],
    )
    dataset.load_data()

    # Clean data (same as final_superset_check in main.py)
    data = dataset.input_data.fillna(0)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    dataset.input_data = data
    dataset.subject_list = data["SubjectID"].unique()

    feature_count = 34  # matches model input_size

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_maes = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset.subject_list)):
        train_subjects = list(dataset.subject_list[train_idx])
        val_subjects = list(dataset.subject_list[val_idx])

        print(f"  Trial {trial.number} | Fold {fold_idx+1}/{N_FOLDS} | "
              f"Train: {len(train_subjects)} | Val: {len(val_subjects)}")

        # Fresh model per fold
        model, optimizer, scheduler, batch_size = create_model_and_optimizer(
            trial, device, feature_count
        )

        fold_mae = train_one_fold(
            model, optimizer, scheduler, dataset,
            train_subjects, val_subjects, batch_size, device, trial, fold_idx
        )
        fold_maes.append(fold_mae)
        print(f"    Fold {fold_idx+1} MAE: {fold_mae:.2f} cm")

    avg_mae = np.mean(fold_maes)
    print(f"  Trial {trial.number} avg MAE: {avg_mae:.2f} cm\n")
    return avg_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing study in DB")
    args = parser.parse_args()

    storage = optuna.storages.RDBStorage(
        url=DB_PATH,
        engine_kwargs={"connect_args": {"timeout": 30}},
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        load_if_exists=True,
    )
    if study.trials:
        print(f"Resuming study with {len(study.trials)} existing trials")

    print(f"Running {args.n_trials} trials...\n")
    study.optimize(objective, n_trials=args.n_trials)

    # Print results
    print("\n" + "=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best avg MAE: {study.best_value:.2f} cm")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best config
    config_path = "models/config/foval_hpo_best.json"
    best = study.best_params
    best["input_size"] = 34
    best["l1_lambda"] = 0.0
    with open(config_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest config saved to {config_path}")


if __name__ == "__main__":
    main()
