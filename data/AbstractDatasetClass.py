from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, MaxAbsScaler, Normalizer, Binarizer, FunctionTransformer
)

from data.foval_preprocessor import createFeatures, subject_wise_normalization, separate_features_and_targets
from data.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset


class AbstractDatasetClass(ABC):
    def __init__(self, data_dir, sequence_length=10):
        self.data_dir = data_dir
        self.input_data = None
        self.subject_list = None
        self.sequence_length = sequence_length
        self.dataset_name = "Unknown"

        # Flags
        self.isGIW = False

        # Scalers and transformers
        self.global_scaler = None
        self.feature_scaler = None
        self.target_scaler = None
        self.best_transformers = None
        self.vergence_depth_range = None
        self.vergence_angle_range = None

        # Column names
        self.target_column_name = 'Gt_Depth'
        self.subject_id_column = 'SubjectID'
        self.multiplicator = 100

        # Depth range
        self.minDepth = 0.35
        self.maxDepth = 3

        # Transformer dictionary for feature optimization
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

        # Scaler configs
        self.scaler_config = {
            'use_minmax': True,
            'use_standard_scaler': False,
            'use_robust_scaler': False,
            'use_quantile_transformer': False,
            'use_power_transformer': False,
            'use_max_abs_scaler': False
        }

    # --- Abstract: must be implemented by subclasses ---

    @abstractmethod
    def load_data(self):
        pass

    # --- Shared pipeline methods ---

    def create_features(self, data_in, is_train=True):
        return createFeatures(data_in, isGIW=self.isGIW, is_train=is_train, dataset=self)

    def normalize_data(self, data, is_train=True):
        features = data.drop(columns=['SubjectID', 'Gt_Depth'])

        if is_train:
            self.global_scaler = RobustScaler()
            normalized = self.global_scaler.fit_transform(features)
        else:
            normalized = self.global_scaler.transform(features)

        data_normalized = pd.DataFrame(normalized, columns=features.columns)
        data_normalized['SubjectID'] = data['SubjectID'].values
        data_normalized['Gt_Depth'] = data['Gt_Depth'].values

        unique_subjects = data_normalized['SubjectID'].unique()
        subject_scaler = RobustScaler()
        return subject_wise_normalization(data_normalized, unique_subjects, subject_scaler)

    def calculate_transformations_for_features(self, data_in):
        best_transformers = {}
        transformed_data = data_in.copy()
        ideal_skew = 0.0
        ideal_kurt = 3.0

        for column in data_in.columns:
            if column in ('Gt_Depth', 'SubjectID'):
                continue

            best_transform = None
            best_transform_name = ""
            min_dist = float('inf')

            for name, transformer_class in self.transformers.items():
                transformer = transformer_class()
                try:
                    data_transformed = transformer.fit_transform(data_in[[column]])
                    current_skew = skew(data_transformed)[0]
                    current_kurt = kurtosis(data_transformed, fisher=False)[0]
                    dist = np.sqrt((current_skew - ideal_skew) ** 2 + (current_kurt - ideal_kurt) ** 2)

                    if dist < min_dist:
                        min_dist = dist
                        best_transform = transformer
                        best_transform_name = name
                except ValueError:
                    continue

            best_transformers[column] = (best_transform_name, best_transform)

            if best_transform:
                transformed_column = best_transform.transform(data_in[[column]])
                transformed_data[column] = transformed_column.squeeze()

        self.best_transformers = best_transformers
        return transformed_data

    def apply_transformations_on_features(self, data_in):
        transformed_data = data_in.copy()

        for column, (name, transformer) in self.best_transformers.items():
            if transformer is None:
                continue
            if column == 'Gt_Depth':
                transformed_data[column] = data_in[[column]]
            elif column == 'SubjectID':
                continue
            else:
                transformed_column = transformer.transform(data_in[[column]])
                transformed_data[column] = transformed_column.squeeze()

        return transformed_data

    def scale_target(self, data_in, isTrain=False):
        gt_depth = data_in['Gt_Depth'].values.reshape(-1, 1)

        if isTrain:
            self.target_scaler = self.select_scaler(self.scaler_config)
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.fit_transform(gt_depth)
        else:
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.transform(gt_depth)

        data_in['Gt_Depth'] = gt_depth.ravel()
        return data_in

    def scale_features(self, data_in, isTrain=True):
        return data_in

    def create_sequences(self, df):
        sequences = []
        grouped_data = df.groupby('SubjectID')
        for subj_id, group in grouped_data:
            for i in range(len(group) - self.sequence_length):
                seq_features = group.iloc[i:i + self.sequence_length].drop(columns=['Gt_Depth', 'SubjectID'])
                seq_target = group.iloc[i + self.sequence_length]['Gt_Depth']
                sequences.append((seq_features, seq_target, subj_id))
        return sequences

    # --- Scaler selection ---

    def select_scaler(self, config):
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

    # --- Data loading ---

    def get_data(self):
        return self.input_data

    def get_data_loader(self, train_index, val_index=None, test_index=None, batch_size=460):
        train_loader = self.prepare_loader(train_index, batch_size, is_train=True)
        val_loader = self.prepare_loader(val_index, batch_size, is_train=False) if val_index is not None else None
        input_size = train_loader.dataset[0][0].shape[1]
        return train_loader, val_loader, input_size

    def prepare_loader(self, subject_index, batch_size, is_train=False):
        subjects = subject_index if isinstance(subject_index, list) else [subject_index]
        data = self.input_data[self.input_data['SubjectID'].isin(subjects)]

        if data.empty:
            raise ValueError(f"No data found for subjects: {subjects}")

        data = self.create_features(data, is_train=is_train)
        data = self.normalize_data(data, is_train=is_train)

        if is_train:
            data = self.calculate_transformations_for_features(data)
        else:
            data = self.apply_transformations_on_features(data)

        data = self.scale_target(data, isTrain=is_train)

        sequences = self.create_sequences(data)
        features, targets = separate_features_and_targets(sequences)
        features_tensor, targets_tensor = create_lstm_tensors_dataset(features, targets)
        return create_dataloaders_dataset(features_tensor, targets_tensor, batch_size=batch_size)

    # --- Utility ---

    @staticmethod
    def restrict_validation_to_training_range(train_data, val_data, target_column='Gt_Depth'):
        train_min = train_data[target_column].min()
        train_max = train_data[target_column].max()
        restricted = val_data[(val_data[target_column] >= train_min) & (val_data[target_column] <= train_max)]
        print(f"Validation data restricted to range: {train_min:.2f} - {train_max:.2f}")
        print(f"Restricted validation dataset size: {len(restricted)} rows")
        return restricted
