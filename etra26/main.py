"""
FOVAL ETRA'26 Evaluation Pipeline

Runs all evaluations automatically:
  A) Each dataset on its own (LOOCV per dataset)
  B) LOOCV over all 44 subjects combined
  C) Cross-dataset evaluation (train on one, validate on another)

Usage:
    python main.py                      # run all evaluations
    python main.py --eval a             # only A
    python main.py --eval b             # only B
    python main.py --eval c             # only C
    python main.py --eval a b           # A and B
    python main.py --model LSTM         # single model type
    python main.py --epochs 100         # override epochs
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from data.foval_preprocessor import input_features
from data.giw_dataset import GIWDataset
from data.robustVision_dataset import RobustVisionDataset
from data.TuftsDataset import TuftsDataset
from data.MixedDatasetClass import MixedDatasetClass
from data.SpecificMixDatasetClass import SpecificMixDatasetClass
from training.foval_trainer import FOVALTrainer

# --- Config ---
DATASETS = {
    'giw': lambda: GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making"),
    'robustvision': lambda: RobustVisionDataset(data_dir="data/input/robustvision/"),
    'tufts': lambda: TuftsDataset(data_dir="data/input/tufts/", test_split_size=10),
}

DATA_DIRS = {
    'giw': 'data/input/gaze_in_wild/',
    'robustvision': 'data/input/robustvision/',
    'tufts': 'data/input/tufts/',
}

RESULTS_FILE = "results/evaluation_results.csv"


def setup_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    return device


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def final_superset_check(dataset):
    data = dataset.input_data.fillna(0)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    dataset.input_data = data
    return dataset


def save_result(eval_type, model_type, train_set, val_set, mae, duration):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Eval', 'Model', 'Train', 'Val', 'MAE_cm', 'Duration_min'])
        writer.writerow([eval_type, model_type, train_set, val_set, f"{mae:.2f}", f"{duration:.1f}"])


# --- A: Each dataset on its own (LOOCV) ---
def a_single_sets(model_type, n_epochs, device):
    print("\n" + "=" * 60)
    print("A) LOOCV per dataset")
    print("=" * 60)

    for name, create_fn in DATASETS.items():
        print(f"\n--- {name.upper()} ---")
        seed_everything()
        dataset = create_fn()
        dataset.load_data()

        trainer = FOVALTrainer(
            config_path="models/config/foval.json", dataset=dataset, device=device,
            feature_names=input_features, save_intermediates_every_epoch=False,
            model_type=model_type)
        trainer.setup()

        t0 = time.time()
        mean_mae = trainer.cross_validate(num_epochs=n_epochs)
        duration = (time.time() - t0) / 60

        print(f"{name.upper()} | {model_type} | MAE: {mean_mae:.2f} cm | {duration:.1f} min")
        save_result('A_single', model_type, name, name, mean_mae, duration)


# --- B: LOOCV over all 44 subjects ---
def b_loocv_all(model_type, n_epochs, device):
    print("\n" + "=" * 60)
    print("B) LOOCV over all 44 subjects")
    print("=" * 60)

    seed_everything()
    combined = MixedDatasetClass(data_dirs=DATA_DIRS)
    combined.load_data()
    combined = final_superset_check(combined)

    trainer = FOVALTrainer(
        config_path="models/config/foval.json", dataset=combined, device=device,
        feature_names=list(combined.input_data.columns),
        save_intermediates_every_epoch=False, model_type=model_type)
    trainer.setup()

    t0 = time.time()
    mean_mae = trainer.cross_validate(num_epochs=n_epochs)
    duration = (time.time() - t0) / 60

    print(f"ALL 44 | {model_type} | MAE: {mean_mae:.2f} cm | {duration:.1f} min")
    save_result('B_loocv_all', model_type, 'all', 'all', mean_mae, duration)


# --- C: Cross-dataset (train on one, validate on another) ---
def c_cross_datasets(model_type, n_epochs, device):
    print("\n" + "=" * 60)
    print("C) Cross-dataset evaluation")
    print("=" * 60)

    dataset_names = list(DATA_DIRS.keys())

    for train_name in dataset_names:
        for val_name in dataset_names:
            if train_name == val_name:
                continue

            print(f"\n--- Train: {train_name.upper()} → Val: {val_name.upper()} ---")
            seed_everything()

            train_dirs = {train_name: DATA_DIRS[train_name]}
            val_dirs = {val_name: DATA_DIRS[val_name]}

            dataset = SpecificMixDatasetClass(train_dirs, val_dirs, None, 10)
            dataset.load_data()
            dataset = final_superset_check(dataset)

            trainer = FOVALTrainer(
                config_path="models/config/foval.json", dataset=dataset, device=device,
                feature_names=list(dataset.train_data.columns),
                save_intermediates_every_epoch=False, model_type=model_type)
            trainer.setup()

            t0 = time.time()
            mean_mae = trainer.cross_validate_with_specific_datasets(num_epochs=n_epochs)
            duration = (time.time() - t0) / 60

            print(f"{train_name}→{val_name} | {model_type} | MAE: {mean_mae:.2f} cm | {duration:.1f} min")
            save_result('C_cross', model_type, train_name, val_name, mean_mae, duration)


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="FOVAL ETRA'26 Evaluation Pipeline")
    parser.add_argument('--eval', nargs='+', default=['a', 'b', 'c'],
                        choices=['a', 'b', 'c'], help='Which evaluations to run')
    parser.add_argument('--model', nargs='+', default=['LSTM'],
                        choices=['LSTM', 'GRU', 'CNN', 'Attention', 'TCN'],
                        help='Model types to evaluate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    args = parser.parse_args()

    seed_everything()
    device = setup_device()

    print(f"Evaluations: {args.eval}")
    print(f"Models: {args.model}")
    print(f"Epochs: {args.epochs}")

    for model_type in args.model:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model_type}")
        print(f"{'#' * 60}")

        if 'a' in args.eval:
            a_single_sets(model_type, args.epochs, device)
        if 'b' in args.eval:
            b_loocv_all(model_type, args.epochs, device)
        if 'c' in args.eval:
            c_cross_datasets(model_type, args.epochs, device)

    print(f"\n{'=' * 60}")
    print(f"All done. Results saved to {RESULTS_FILE}")
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            print(f.read())


if __name__ == "__main__":
    main()
