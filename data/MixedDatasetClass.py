import numpy as np
import pandas as pd

from data.AbstractDatasetClass import AbstractDatasetClass
from data.TuftsDataset import TuftsDataset
from data.foval_preprocessor import input_features, separate_features_and_targets
from data.giw_dataset import GIWDataset
from data.robustVision_dataset import RobustVisionDataset
from data.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset


class MixedDatasetClass(AbstractDatasetClass):
    def __init__(self, data_dirs):
        super().__init__(data_dir=None, sequence_length=10)
        self.data_dirs = data_dirs
        self.dataset_name = "Mixed"

    def load_data(self):
        print("Reading and aggregating data...")

        robustvision_dataset = RobustVisionDataset(data_dir="data/input/robustvision/")
        robustvision_dataset.load_data()
        robustvision_data = robustvision_dataset.input_data
        robustvision_dataset.create_features(robustvision_data)

        giw_dataset = GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making")
        giw_dataset.load_data()
        giw_data = giw_dataset.input_data
        giw_dataset.create_features(giw_data)

        tufts_dataset = TuftsDataset(data_dir="data/input/tufts/", test_split_size=10)
        tufts_dataset.load_data()
        tufts_data = tufts_dataset.input_data
        tufts_dataset.create_features(tufts_data)

        self.input_data = pd.concat([robustvision_data, giw_data, tufts_data], ignore_index=True)
        self.subject_list = pd.unique(pd.concat([
            pd.Series(robustvision_dataset.subject_list),
            pd.Series(tufts_dataset.subject_list),
            pd.Series(giw_dataset.subject_list)
        ]))

        self.input_data = self.input_data[input_features]

        print(f"Finished aggregating data.")
        print(f"Total subjects: {len(self.subject_list)}")
        print(f"Total data points: {len(self.input_data)}")

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
