import pickle

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


class MixedDatasetClass(AbstractDatasetClass):
    def __init__(self, data_dirs):
        """
        Initialize the RobustVisionDataset class.

        """
        self.data_dirs = data_dirs
        self.input_data = None
        self.subject_list = None
        self.sequence_length = 10  # sequence_length
        self.best_transformers = None
        self.minDepth = 0.35  # in meter
        self.maxDepth = 3
        self.subject_scaler = RobustScaler()  # or any other scaler
        self.feature_scaler = None
        self.target_scaler = None
        self.target_column_name = 'Gt_Depth'
        self.subject_id_column = 'SubjectID'
        self.multiplicator = 100  # to convert from cm to m
        self.transformers = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'MaxAbsScaler': MaxAbsScaler,
            'RobustScaler': RobustScaler,
            'QuantileTransformer-Normal': lambda: QuantileTransformer(output_distribution='normal'),
            'QuantileTransformer-Uniform': lambda: QuantileTransformer(output_distribution='uniform'),
            'PowerTransformer-YeoJohnson': lambda: PowerTransformer(method='yeo-johnson'),
            'PowerTransformer-BoxCox': lambda: PowerTransformer(method='box-cox'),
            'Normalizer': Normalizer,
            'Binarizer': lambda threshold=0.0: Binarizer(threshold=threshold),
            'FunctionTransformer-logp1p': lambda func=np.log1p: FunctionTransformer(func),
            'FunctionTransformer-rec': lambda func=np.reciprocal: FunctionTransformer(func),
            'FunctionTransformer-sqrt': lambda func=np.sqrt: FunctionTransformer(func),
        }

        self.scaler_config = {
            'use_minmax': True,
            'use_standard_scaler': False,
            'use_robust_scaler': False,
            'use_quantile_transformer': False,
            'use_power_transformer': False,
            'use_max_abs_scaler': False
        }

        self.scaler_config_features = {
            'use_minmax': False,  # avg 56
            'use_standard_scaler': False,  # avg 15
            'use_robust_scaler': False,  # avg 15.5
            'use_quantile_transformer': False,  # avg 14.4
            'use_power_transformer': False,  # avg 15.2
            'use_max_abs_scaler': False  # avg 10.79
            # none                               # avg
        }


    def load_data_new(self):
        print("Loading and combining datasets...")
        combined_data = []

        for dataset_name, dataset_path in self.data_dirs.items():
            print(f"Processing dataset: {dataset_name}")
            dataset_data = self._load_individual_dataset(dataset_name, dataset_path)
            combined_data.append(dataset_data)

        self.input_data = pd.concat(combined_data, ignore_index=True)
        self.subject_list = self.input_data['SubjectID'].unique()
        print(f"Successfully loaded datasets with {len(self.input_data)} rows.")

    def _load_individual_dataset(self, dataset_name, dataset_path):
        print(f"Loading data from {dataset_path}...")
        data_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        dataset_data = []

        for file in data_files:
            file_path = os.path.join(dataset_path, file)
            print(f"Processing file: {file}")

            data = pd.read_csv(file_path)
            data['SubjectID'] = dataset_name + "_" + file.split('.')[0]
 
            dataset_data.append(data)

        return pd.concat(dataset_data, ignore_index=True)
    
    # 1.
    def load_data(self):
        """
        Read and aggregate data from multiple subjects.

        :param data_dir: Directory containing the subject folders.
        :return: Combined dataframe of all subjects.
        """
        print("Reading and aggregating data...")

        # Robust Vision Dataset
        robustvision_dataset = RobustVisionDataset(data_dir="data/input/robustvision/")
        robustvision_dataset.load_data()
        robustvision_data = robustvision_dataset.input_data
        robustvision_dataset.create_features(robustvision_data)
        rv_subjects = robustvision_dataset.subject_list

        # GIW Dataset
        giw_dataset = GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making")
        giw_dataset.load_data()
        giw_data = giw_dataset.input_data
        giw_dataset.create_features(giw_data)
        giw_subjects = giw_dataset.subject_list

        # Tufts Dataset
        tufts_dataset = TuftsDataset(data_dir="data/input/tufts/", test_split_size=10)
        tufts_dataset.load_data()
        tufts_data = tufts_dataset.input_data
        tufts_dataset.create_features(tufts_data)
        tufts_subjects = tufts_dataset.subject_list

        # Combine input_data from all datasets
        self.input_data = pd.concat([robustvision_data, giw_data, tufts_data], ignore_index=True)
        # self.input_data = pd.concat([tufts_data, robustvision_data], ignore_index=True)


        # Combine subjects from all datasets
        self.subject_list = pd.unique(pd.concat([pd.Series(rv_subjects),
                                                 pd.Series(tufts_subjects),
                                                 pd.Series(giw_subjects)
                                                 ]))

        self.input_data = self.input_data[input_features]


        print("Finished aggregating data.")
        print(f"Total subjects: {len(self.subject_list)}")
        print(f"Total data points: {len(self.input_data)}")

        # Beispiel-Daten
        datasets = {
            'robustvision': robustvision_data,
            'tufts': tufts_data,
            'giw': giw_data
        }

        self.analyze_target_distribution(datasets)

        # Robust vs Tufts


    def analyze_target_distribution(self, dataframes, target_column='Gt_Depth'):
        """
        Analyzes and visualizes the range and distribution of the target variable in each dataset.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        for name, data in dataframes.items():
            plt.hist(data[target_column], bins=50, alpha=0.5, label=f'{name} (Range: {data[target_column].min()} - {data[target_column].max()})')

        plt.title('Target Variable Distribution Across Datasets')
        plt.xlabel('Gt_Depth (meters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        for name, data in dataframes.items():
            min_val = data[target_column].min()
            max_val = data[target_column].max()
            print(f"{name}: Gt_Depth range = {min_val:.2f} - {max_val:.2f} meters")

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

    # Utilities
    def select_scaler(self, config):
        """Select the scaler based on the configuration provided."""
        if config['use_minmax']:
            return MinMaxScaler(feature_range=(0, 1000))
        if config['use_standard_scaler']:
            return StandardScaler()
        if config['use_robust_scaler']:
            return RobustScaler(with_scaling=True, with_centering=True, unit_variance=True)
        if config['use_quantile_transformer']:
            return QuantileTransformer(output_distribution='normal')
        if config['use_power_transformer']:
            return PowerTransformer(method='yeo-johnson')
        if config['use_max_abs_scaler']:
            return MaxAbsScaler()
        return None

    def get_data(self):
        return self.input_data

    # def get_data_loader(self, train_index, val_index=None, test_index=None, batch_size=100):
    #     """
    #     Create and return data loaders for training, validation, and testing datasets.

    #     :param train_index: Indices for training subjects.
    #     :param val_index: Indices for validation subjects (optional).
    #     :param test_index: Indices for test subjects (optional).
    #     :param batch_size: Batch size for the data loaders.
    #     :return: Data loaders for training, validation, and testing datasets, and the input size.
    #     """
    #     train_loader = self.prepare_loader(train_index, batch_size, is_train=True)
    #     print("vals", val_index)
    #     val_loader = self.prepare_loader(val_index, batch_size, is_train=False) if val_index is not None else None
    #     # test_loader = self.prepare_loader(test_index, batch_size, is_train=False) if test_index is not None else None

    #     input_size = train_loader.dataset[0][0].shape[1]  # Assuming the first dimension is batch_size

    #     return train_loader, val_loader, input_size

    # def prepare_loader(self, subject_index, batch_size, is_train=False):
    #     print("vals2", subject_index)

    #    # Sicherstellen, dass subject_index eine Liste ist
    #     if isinstance(subject_index, str):
    #         subject_index = [subject_index]
    #     elif not isinstance(subject_index, list):
    #         raise ValueError(f"subject_index muss eine Liste oder ein String sein, erhalten: {type(subject_index)}")

    #     # Debugging-Ausgabe, um den Typ zu überprüfen
    #     print(f"subject_index Typ: {type(subject_index)}, Wert: {subject_index}")

    #     # Daten filtern
    #     data = self.input_data[self.input_data['SubjectID'].isin(subject_index)]
    #     # if is_train:
    #     #     data.to_csv('checkpoint_raw_1.csv')

    #     # Check if the data is empty before proceeding
    #     if data.empty:
    #         raise ValueError(f"No data found for subjects: {subject_index}")

    #     # Feature creation and normalization
    #     # data = self.create_features(data)
    #     # if is_train:
    #     #     data.to_csv('checkpoint_features_2.csv')

    #     data = self.normalize_data(data)

    #     # Apply transformations if necessary
    #     if is_train:
    #         #     data.to_csv('checkpoint_normalized_3.csv')

    #         data = self.calculate_transformations_for_features(data)
    #         # data.to_csv('checkpoint_transformed_4.csv')

    #     else:
    #         data = self.apply_transformations_on_features(data)

    #     # Scale features and target (transform only using the fitted scaler)
    #     # data = self.scale_features(data, isTrain=is_train)
    #     data = self.scale_target(data, isTrain=is_train)
    #     # if is_train:
    #     #     data.to_csv('checkpoint_scaled_5.csv')

    #     # Generate sequences
    #     sequences = self.create_sequences(data)
    #     features, targets = separate_features_and_targets(sequences)

    #     # Convert to tensors and create data loader
    #     features_tensor, targets_tensor = create_lstm_tensors_dataset(features, targets)
    #     data_loader = create_dataloaders_dataset(features_tensor, targets_tensor, batch_size=batch_size)

    #     # if is_train:
    #     #     # Assuming your list is called sequences
    #     #     with open('train_sequences_new.pkl', 'wb') as f:
    #     #         pickle.dump(sequences, f)

    #     # sequences.to_pickle("train_sequences_new.pkl")

    #     return data_loader

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