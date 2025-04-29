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

from tqdm import tqdm

from data.AbstractDatasetClass import AbstractDatasetClass
from data.foval_preprocessor import remove_outliers_in_labels, binData, createFeatures, \
    detect_and_remove_outliers_in_features_iqr, clean_data, global_normalization, subject_wise_normalization, \
    separate_features_and_targets
from data.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset
from data.utilities import create_lstm_tensors_dataset, \
    create_dataloaders_dataset
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobustVisionDataset(AbstractDatasetClass):
    def __init__(self, data_dir, sequence_length):
        """
        Initialize the RobustVisionDataset class.

        """
        super().__init__(data_dir, sequence_length)
        self.dataset_name = "robustvision"
        self.input_data = None
        self.subject_list = None
        self.sequence_length = 10  # sequence_length
        self.data_dir = data_dir
        self.best_transformers = None
        self.minDepth = 0.35  # in meter
        self.maxDepth = 3
        self.subject_scaler = RobustScaler()  # or any other scaler
        self.feature_scaler = None
        self.isGIW= False # für mixed muss das auf True stehen sonst False

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

    # 1.
    def load_data(self):
        """
        Read and aggregate data from multiple subjects.

        :param data_dir: Directory containing the subject folders.
        :return: Combined dataframe of all subjects.
        """
        print("Reading and aggregating data...")
        all_data = []
        for subj_folder in os.listdir(self.data_dir):
            subj_path = os.path.join(self.data_dir, subj_folder)
            if os.path.exists(subj_path):
                depthCalib_path = os.path.join(subj_path, "depthCalibration.csv")
                if os.path.exists(depthCalib_path):
                    df = pd.read_csv(depthCalib_path, delimiter="\t")
                    df.rename(columns=lambda x: x.strip().replace(' ', '_').title(), inplace=True)
                    # df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

                    starting_columns = [
                        'Gt_Depth', 'World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y',
                        'World_Gaze_Direction_L_Z', 'World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y',
                        'World_Gaze_Direction_R_Z', 'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Z',
                        'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Z'
                    ]

                    # self.clean_column_names(df_depthEval)
                    df = df[starting_columns]
                    for col in starting_columns:
                        df[col] = df[col].astype(float)

                    df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

                    df = df[starting_columns].astype(float)
                    df2 = clean_data(df, target_column_name=self.target_column_name, multiplication=self.multiplicator)
                    df3 = remove_outliers_in_labels(df2, window_size=5, threshold=10,
                                                    target_column_name=self.target_column_name)
                    df4 = detect_and_remove_outliers_in_features_iqr(df3)
                    df5 = binData(df4, False)
                    df5['SubjectID'] = subj_folder
                    all_data.append(df5)

        self.input_data = pd.concat(all_data, ignore_index=True)
        print("Finished loading input data.")
        self.subject_list = self.input_data['SubjectID'].unique()

        return self.input_data


    # 2.
    def create_features(self, data_in):
        """
        Generate features from the input data.

        :param data_in: Input dataframe.
        :return: Dataframe with additional features.
        @param data_in:
        @return:
        """
        data_in = createFeatures(data_in, isGIW=self.isGIW)

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

    def get_data_loader(self, train_index, val_index=None, test_index=None, batch_size=460, feature_transformer=None):
        """
        Create and return data loaders for training, validation, and testing datasets.
        """
        train_loader = self.prepare_loader(train_index, batch_size=batch_size, is_train=True,
                                           feature_transformer=feature_transformer)

        val_loader = None
        if val_index is not None:
            val_loader = self.prepare_loader(val_index, batch_size=batch_size, is_train=False,
                                             feature_transformer=feature_transformer)

        test_loader = None
        if test_index is not None:
            test_loader = self.prepare_loader(test_index, batch_size=batch_size, is_train=False,
                                              feature_transformer=feature_transformer)

        input_size = train_loader.dataset[0][0].shape[1]  # [batch, time, features]
        return train_loader, val_loader, input_size

    def prepare_loader(self, subject_index, batch_size=460, is_train=False, feature_transformer=None):
        """
        Load preprocessed subject data (.pkl), apply scaling, transformation, sequence creation.
        """


        print(f"Preparing data loader... is_train={is_train}")
        subjects = subject_index if isinstance(subject_index, list) else [subject_index]
        print(f"Subjects to prepare: {subjects}")

        all_features = []
        all_targets = []

        for subject_id in subjects:
            pkl_path = os.path.join("cached_subjects", f"{subject_id}.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"❌ Cached file for subject {subject_id} not found at {pkl_path}")

            data = pd.read_pickle(pkl_path)

            if data.empty:
                print(f"⚠️ Skipping empty subject {subject_id}")
                continue

            # Transformationen und Skalierung
            if is_train:
                data = self.calculate_transformations_for_features(data)
            else:
                data = self.apply_transformations_on_features(data)

            # data = self.scale_features(data, isTrain=is_train)
            data = self.scale_target(data, isTrain=is_train)

            # Sequenzen erzeugen
            sequences = self.create_sequences(data)
            features, targets = separate_features_and_targets(sequences)

            all_features.append(features)
            all_targets.append(targets)

        # Alle Subjekte zusammenfassen
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_targets, axis=0)

        # In Tensor-Loader umwandeln
        X_tensor, y_tensor = create_lstm_tensors_dataset(X, y)
        loader = create_dataloaders_dataset(X_tensor, y_tensor, batch_size=batch_size, shuffle=is_train)

        return loader

    def prepare_loader_with_fit(self, train_subjects, val_subjects, batch_size=460):
        # 1. Lade train pickle, fit transformer, apply transform, make sequences
        train_X, train_y = self._load_and_process_subjects(train_subjects, is_train=True)

        # 2. Lade val pickle, apply transformer, make sequences
        val_X, val_y = self._load_and_process_subjects(val_subjects, is_train=False)

        # 3. Erstelle DataLoader
        train_loader = create_dataloaders_dataset(*create_lstm_tensors_dataset(train_X, train_y), batch_size=batch_size,
                                                  shuffle=True)
        val_loader = create_dataloaders_dataset(*create_lstm_tensors_dataset(val_X, val_y), batch_size=batch_size,
                                                shuffle=False)

        input_size = train_X.shape[-1]
        return train_loader, val_loader, input_size

    def _load_and_process_subjects(self, subjects, is_train):
        all_features, all_targets = [], []
        for subject in subjects:
            data = pd.read_pickle(f"cached_subjects/{subject}.pkl")
            if is_train:
                data = self.calculate_transformations_for_features(data)
                data = self.scale_target(data, isTrain=True)
            else:
                data = self.apply_transformations_on_features(data)
                data = self.scale_target(data, isTrain=False)

            sequences = self.create_sequences(data)
            X, y = separate_features_and_targets(sequences)
            all_features.append(X)
            all_targets.append(y)

        return np.concatenate(all_features), np.concatenate(all_targets)

    def preprocess_all_subjects_once(self, output_dir="cached_subjects_raw"):
        """
        Preprocess all subjects once (only feature creation and normalization),
        and save raw sequences as .npz files (no scaling or transformations).
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"📦 Preprocessing and caching subjects to: {output_dir}")

        for subject_id in tqdm(self.subject_list):
            try:
                print(f"→ Preprocessing Subject: {subject_id}")
                data = self.input_data[self.input_data['SubjectID'] == subject_id].copy()

                if data.empty:
                    print(f"⚠️  Skipping {subject_id} (no data)")
                    continue

                # Nur subjektbasierte Schritte
                data = self.create_features(data)
                data = self.normalize_data(data)

                # Noch keine Transformationen, keine Skalierung, keine Sequenzen
                # Aber trotzdem speichern: als DataFrame im Pickle-Format
                save_path = os.path.join(output_dir, f"{subject_id}.pkl")
                data.to_pickle(save_path)

            except Exception as e:
                print(f"❌ Error processing subject {subject_id}: {e}")
