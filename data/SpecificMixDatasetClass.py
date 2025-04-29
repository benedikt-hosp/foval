import pickle

import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import (
    PowerTransformer, Normalizer, MaxAbsScaler, RobustScaler,
    QuantileTransformer, StandardScaler, MinMaxScaler,
    FunctionTransformer, Binarizer
)
import warnings

from data.AbstractDatasetClass import AbstractDatasetClass
from data.TuftsDataset import TuftsDataset
from data.foval_preprocessor import remove_outliers_in_labels, binData, createFeatures, \
    detect_and_remove_outliers_in_features_iqr, clean_data, global_normalization, subject_wise_normalization, \
    separate_features_and_targets, input_features
from data.giw_dataset import GIWDataset
from data.robustVision_dataset import RobustVisionDataset
from data.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpecificMixDatasetClass(AbstractDatasetClass):
    def __init__(self, train_data_dirs, val_data_dirs, data_dir, sequence_length):
        """
        Initialize with separate directories for training and validation datasets.
        """
        super().__init__(data_dir, sequence_length)
        self.train_data_dirs = train_data_dirs
        self.val_data_dirs = val_data_dirs
        self.train_data = train_data_dirs
        self.val_data = val_data_dirs

    def load_data(self):
        """
        Load and combine training and validation datasets separately.
        """
        print("Loading training data...")
        if len(self.train_data_dirs.items()) > 1:
            self.train_data = self._load_multiple_datasets(self.train_data_dirs)
        else:
            self.train_data = self._load_individual_dataset(list(self.train_data_dirs.keys())[0], list(self.train_data_dirs.values())[0])


        print("Loading validation data...")
        self.val_data = self._load_multiple_datasets(self.val_data_dirs)
        # self.val_data = self._load_individual_dataset("tufts", "data/input/tufts/")

        # Optionally combine both datasets for overall reference
        self.input_data = pd.concat([self.train_data, self.val_data], ignore_index=True)

        # Handle NaN and infinite values
        numerical_columns =  self.input_data.select_dtypes(include=[np.number])
        combined_data =  self.input_data.fillna(0)
        numerical_columns.replace([np.inf, -np.inf], 0, inplace=True)

        # Debugging output
        print("Final NaN values:", numerical_columns.isna().sum().sum())
        print("Final Inf values:", np.isinf(numerical_columns).sum().sum())

            
        # Drop NaN values created by diff() function
        combined_data = combined_data.dropna()

        combined_data = combined_data.replace([np.inf, -np.inf], np.nan)
        combined_data = combined_data.dropna()
        # Define excluded features
        # excluded_features = ['World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Y', 'World_Gaze_Origin_R_Z', 
        #                     'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Y', 'World_Gaze_Origin_L_Z']

        # # Remove excluded features
        # combined_data = combined_data.drop(columns=excluded_features)

        print("Combined Data:")
        print( combined_data.head())

        self.input_data = combined_data[input_features]

        # print("Combined Subjects:")
        # print( self.input_data.subject_list)


        print(f"Training data size: {len(self.train_data)} rows.")
        print(f"Validation data size: {len(self.val_data)} rows.")


    def _load_individual_dataset(self, dataset_name, dataset_path):
        print(f"Loading dataset {dataset_name}")
        print(f"Loading data from {dataset_path}...")


        dataset_data = []
        # Robust Vision Dataset
        if dataset_name == "robustvision":
            robustvision_dataset = RobustVisionDataset(data_dir="data/input/robustvision/")
            robustvision_dataset.load_data()
            robustvision_data = robustvision_dataset.input_data
            robustvision_dataset.create_features(robustvision_data)
            # rv_subjects = robustvision_dataset.subject_list
            # dataset_data.append(robustvision_data)
            return robustvision_data


        # GIW Dataset
        if dataset_name == "giw":
            giw_dataset = GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making")
            giw_dataset.load_data()
            giw_data = giw_dataset.input_data
            giw_dataset.create_features(giw_data)
            # giw_subjects = giw_dataset.subject_list
            # dataset_data.append(giw_dataset)
            return giw_data


        # Tufts Dataset
        if dataset_name == "tufts":
            tufts_dataset = TuftsDataset(data_dir="data/input/tufts/", test_split_size=10)
            tufts_dataset.load_data()
            tufts_data = tufts_dataset.input_data
            tufts_dataset.create_features(tufts_data)
            # tufts_subjects = tufts_dataset.subject_list
            return tufts_data
        
    
    def _load_multiple_datasets(self, data_dirs):
        """
        Load and combine multiple datasets from specified directories.
        """
        combined_data = []
        for dataset_name, dataset_path in data_dirs.items():
            print(f"Processing dataset: {dataset_name}")
            dataset_data = self._load_individual_dataset(dataset_name, dataset_path)
            combined_data.append(dataset_data)
        
        return pd.concat(combined_data, ignore_index=True)

    def get_data_loader(self, train_subjects=None, val_subjects=None, batch_size=460):
        """
        Prepare data loaders for training and validation datasets.
        """

        train_subjects_checked = train_subjects if isinstance(train_subjects, list) else [train_subjects]
        train_data = self.input_data[self.input_data['SubjectID'].isin(train_subjects_checked)]

        val_subjects_checked = val_subjects if isinstance(val_subjects, list) else [val_subjects]
        val_data = self.input_data[self.input_data['SubjectID'].isin(val_subjects_checked)]

          # Restrict validation data to training range
        val_data = self.restrict_validation_to_training_range(train_data, val_data, target_column='Gt_Depth')

        train_loader = self.prepare_loader(train_subjects_checked, batch_size, data=train_data, is_train=True)
        val_loader = self.prepare_loader(val_subjects, batch_size, data=val_data, is_train=False)

        input_size = train_loader.dataset[0][0].shape[1]  # Assuming the first dimension is batch_size

        print("Data loaders prepared.")
        return train_loader, val_loader, input_size
    
    
    def restrict_validation_to_training_range(self, train_data, val_data, target_column='Gt_Depth'):
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

    def prepare_loader(self, subject_index, batch_size, data, is_train=True):
        """
        Prepare a data loader for a given subset of subjects.
        """    
        
    
        # Check if the data is empty before proceeding
        if data.empty:
            raise ValueError(f"No data found for subjects: {subject_index}")

        # data = self.normalize_data(data)

        # Apply transformations if necessary
        # if is_train:
        #     data = self.calculate_transformations_for_features(data)
            

        # else:
        #     data = self.apply_transformations_on_features(data)

        # Scale features and target (transform only using the fitted scaler)
        # data = self.scale_features(data, isTrain=is_train)
        data = self.scale_target(data, isTrain=is_train)
        # if is_train:
        #     data.to_csv('checkpoint_scaled_5.csv')

        # Generate sequences
        sequences = self.create_sequences(data)
        features, targets = separate_features_and_targets(sequences)

        # Convert to tensors and create data loader
        features_tensor, targets_tensor = create_lstm_tensors_dataset(features, targets)
        data_loader = create_dataloaders_dataset(features_tensor, targets_tensor, batch_size=batch_size)

        # if is_train:
        #     # Assuming your list is called sequences
        #     with open('train_sequences_new.pkl', 'wb') as f:
        #         pickle.dump(sequences, f)

        # sequences.to_pickle("train_sequences_new.pkl")

        return data_loader


    # 2.
    def create_features(self, data_in):
        """
        Generate features from the input data.

        :param data_in: Input dataframe.
        :return: Dataframe with additional features.
        @param data_in:
        @return:
        """
        data_in = createFeatures(data_in, isGIW=False)

        return data_in

    # 3.
    def normalize_data(self, data):

        # Apply global normalization first
        data_set_in = global_normalization(data)

        # Then proceed with the existing subject-wise normalization
        unique_subjects = data_set_in['SubjectID'].unique()

        # Choose your scaler for subject-wise normalization
        subject_scaler = RobustScaler()

        # Apply subject-wise normalization
        dataset_in_normalized = subject_wise_normalization(data_set_in, unique_subjects, subject_scaler)

        return dataset_in_normalized

    # 4. A (Traininig data)
    def calculate_transformations_for_features(self, data_in):
        best_transformers = {}
        transformed_data = data_in.copy()
        ideal_skew = 0.0
        ideal_kurt = 3.0

        for column in data_in.columns:
            if column == "Gt_Depth" or column == "SubjectID":
                continue

            # print(f"Processing column: {column}")
            original_skew = skew(data_in[column])
            original_kurt = kurtosis(data_in[column], fisher=False)  # Pearson's definition

            best_transform = None
            best_transform_name = ""
            min_skew_diff = float('inf')

            for name, transformer_class in self.transformers.items():
                transformer = transformer_class()  # Create a new object for each transformer
                try:
                    data_transformed = transformer.fit_transform(data_in[[column]])
                    current_skew = skew(data_transformed)[0]
                    current_kurt = kurtosis(data_transformed, fisher=False)[0]

                    # Calculate the distance from the ideal distribution characteristics
                    dist = np.sqrt((current_skew - ideal_skew) ** 2 + (current_kurt - ideal_kurt) ** 2)

                    # If this transformer is the best so far, store it
                    if dist < min_skew_diff:
                        min_skew_diff = dist
                        best_transform = transformer
                        best_transform_name = name

                except ValueError as e:  # Handle failed transformations, e.g., Box-Cox with negative values
                    # print(f"Transformation failed for {name} on column {column}: {e}")
                    continue

            best_transformers[column] = (best_transform_name, best_transform)

            # Transform the column in the dataset
            if best_transform:
                transformed_column = best_transform.transform(data_in[[column]])
                transformed_data[column] = transformed_column.squeeze()

        self.best_transformers = best_transformers
        return transformed_data

    # 4. B ( Validation/ test data)
    def apply_transformations_on_features(self, data_in):
        transformed_validation_data = data_in.copy()

        for column, (name, transformer) in self.best_transformers.items():
            if transformer is not None:
                if column == "Gt_Depth":
                    transformed_validation_data[column] = data_in[[column]]
                elif column == "SubjectID":
                    continue
                else:
                    # Apply the transformation using the fitted transformer object
                    transformed_column = transformer.transform(data_in[[column]])
                    transformed_validation_data[column] = transformed_column.squeeze()

        return transformed_validation_data
    
    
    # 5.
    def scale_target(self, data_in, isTrain=False):

        if isTrain:
            self.target_scaler = self.select_scaler(self.scaler_config)
            # Extract GT_depth before scaling and reshape for scaler compatibility
            gt_depth = data_in['Gt_Depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.fit_transform(gt_depth)
                # Re-attach the excluded columns
            data_in['Gt_Depth'] = gt_depth.ravel()
        else:
            gt_depth = data_in['Gt_Depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.transform(gt_depth)
                # Re-attach the excluded columns
            data_in['Gt_Depth'] = gt_depth.ravel()

        return data_in

    # 6.
    def scale_features(self, data_in, isTrain=True):
        """
        Scale the features in the training and validation datasets using the provided scaler.

        :param data_in: Dataframe with training data.
        :return: Scaled dataframe.
        @param isTrain:
        """

        # target_column = data_in[self.target_column_name].values.reshape(-1, 1)
        # subject_id_column = data_in[self.subject_id_column]
        # data = data_in.drop(columns=[self.target_column_name, self.subject_id_column])
        #
        # # Fit the scaler on the training data
        # if isTrain:
        #     self.feature_scaler = self.select_scaler(self.scaler_config_features)
        #     # Fit the scaler only if it's not already fitted (e.g., during validation)
        #     data_scaled = self.feature_scaler.fit_transform(data)
        # else:
        #     data_scaled = self.feature_scaler.transform(data)
        #
        # data_out = pd.DataFrame(data_scaled, columns=data.columns)
        # data_out[self.target_column_name] = target_column.ravel()
        # data_out[self.subject_id_column] = subject_id_column.reset_index(drop=True)

        return data_in

    # 6.
    def create_sequences(self, df):
        """
        Create sequences of data for time-series analysis.

        :param df: Dataframe with the data to sequence.
        :return: List of sequences, where each sequence is a tuple of (features, target, subject_id).
        """
        sequences = []
        grouped_data = df.groupby('SubjectID')
        for subj_id, group in grouped_data:
            for i in range(len(group) - self.sequence_length):
                seq_features = group.iloc[i:i + self.sequence_length].drop(columns=['Gt_Depth', 'SubjectID'])
                seq_target = group.iloc[i + self.sequence_length]['Gt_Depth']
                sequences.append((seq_features, seq_target, subj_id))
        return sequences
