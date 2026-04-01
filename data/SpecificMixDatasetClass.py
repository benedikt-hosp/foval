import numpy as np
import pandas as pd

from data.AbstractDatasetClass import AbstractDatasetClass
from data.TuftsDataset import TuftsDataset
from data.foval_preprocessor import input_features, separate_features_and_targets
from data.giw_dataset import GIWDataset
from data.robustVision_dataset import RobustVisionDataset
from data.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset


class SpecificMixDatasetClass(AbstractDatasetClass):
    def __init__(self, train_data_dirs, val_data_dirs, data_dir, sequence_length):
        super().__init__(data_dir, sequence_length)
        self.train_data_dirs = train_data_dirs
        self.val_data_dirs = val_data_dirs
        self.train_data = None
        self.val_data = None

    def load_data(self):
        print("Loading training data...")
        if len(self.train_data_dirs) > 1:
            self.train_data = self._load_multiple_datasets(self.train_data_dirs)
        else:
            name = list(self.train_data_dirs.keys())[0]
            path = list(self.train_data_dirs.values())[0]
            self.train_data = self._load_individual_dataset(name, path)

        print("Loading validation data...")
        self.val_data = self._load_multiple_datasets(self.val_data_dirs)

        self.input_data = pd.concat([self.train_data, self.val_data], ignore_index=True)

        # Clean NaN and infinite values
        self.input_data = self.input_data.fillna(0)
        self.input_data = self.input_data.replace([np.inf, -np.inf], np.nan).dropna()
        self.input_data = self.input_data[input_features]

        print(f"Training data size: {len(self.train_data)} rows.")
        print(f"Validation data size: {len(self.val_data)} rows.")

    def _load_individual_dataset(self, dataset_name, dataset_path):
        print(f"Loading dataset {dataset_name} from {dataset_path}...")

        dataset_map = {
            'robustvision': lambda: RobustVisionDataset(data_dir="data/input/robustvision/"),
            'giw': lambda: GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making"),
            'tufts': lambda: TuftsDataset(data_dir="data/input/tufts/", test_split_size=10),
        }

        if dataset_name not in dataset_map:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset = dataset_map[dataset_name]()
        dataset.load_data()
        dataset.create_features(dataset.input_data)
        return dataset.input_data

    def _load_multiple_datasets(self, data_dirs):
        combined = []
        for name, path in data_dirs.items():
            combined.append(self._load_individual_dataset(name, path))
        return pd.concat(combined, ignore_index=True)

    def get_data_loader(self, train_subjects=None, val_subjects=None, batch_size=460):
        train_subjects = train_subjects if isinstance(train_subjects, list) else [train_subjects]
        val_subjects = val_subjects if isinstance(val_subjects, list) else [val_subjects]

        train_data = self.input_data[self.input_data['SubjectID'].isin(train_subjects)]
        val_data = self.input_data[self.input_data['SubjectID'].isin(val_subjects)]
        val_data = self.restrict_validation_to_training_range(train_data, val_data)

        train_loader = self._prepare_from_data(train_data, batch_size, is_train=True)
        val_loader = self._prepare_from_data(val_data, batch_size, is_train=False)

        input_size = train_loader.dataset[0][0].shape[1]
        return train_loader, val_loader, input_size

    def prepare_loader(self, subject_index, batch_size, is_train=False):
        subjects = subject_index if isinstance(subject_index, list) else [subject_index]
        data = self.input_data[self.input_data['SubjectID'].isin(subjects)]
        return self._prepare_from_data(data, batch_size, is_train=is_train)

    def _prepare_from_data(self, data, batch_size, is_train=True):
        if data.empty:
            raise ValueError("No data found for the given subjects.")

        data = self.scale_target(data, isTrain=is_train)

        sequences = self.create_sequences(data)
        features, targets = separate_features_and_targets(sequences)
        features_tensor, targets_tensor = create_lstm_tensors_dataset(features, targets)
        return create_dataloaders_dataset(features_tensor, targets_tensor, batch_size=batch_size)
