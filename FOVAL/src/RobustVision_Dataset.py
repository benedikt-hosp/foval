import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer, Normalizer, MaxAbsScaler, RobustScaler, QuantileTransformer, \
    StandardScaler, MinMaxScaler, FunctionTransformer, Binarizer
import warnings
from FOVAL_Preprocessor import detect_and_remove_outliers, binData, createFeatures, \
    detect_and_remove_outliers_in_features_iqr, selected_features

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.option_context('mode.use_inf_as_na', True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_cleaning(df):
    df = df.dropna(how='all')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().copy()

    df2 = df[df['GT depth'] > 0.35]
    df2 = df2[df2['GT depth'] <= 3]
    df2["GT depth"] = df2["GT depth"].multiply(100)
    df2 = df2.reset_index(drop=True)

    # Detect outliers
    df3 = detect_and_remove_outliers(df2, window_size=10, threshold=30)  # 5, 100
    df4 = detect_and_remove_outliers_in_features_iqr(df3)
    df4 = binData(df4)
    return df4


def create_sequences(df, sequence_length=10):
    sequences = []
    # Group data by subject
    grouped_data = df.groupby('SubjectID')
    for subj_id, group in grouped_data:
        for i in range(len(group) - sequence_length):
            seq_features = group.iloc[i:i + sequence_length].drop(columns=['GT depth', 'SubjectID'])
            seq_target = group.iloc[i + sequence_length]['GT depth']
            sequences.append((seq_features, seq_target, subj_id))
    return sequences


def create_features(data_in):
    data_in = createFeatures(data_in)
    print(data_in.columns)
    # If you want to only use specific features, you can filter them here
    # selected_features = ..
    #data_in = data_in[selected_features]
    return data_in


class RobustVision_Dataset:
    pd.option_context('mode.use_inf_as_na', True)

    def __init__(self, sequence_length):
        self.best_transformers = None
        self.feature_scaler = None
        self.target_scaler = None
        self.train_targets = None
        self.valid_features = None
        self.valid_targets = None
        self.train_features = None
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

        self.sequence_length = sequence_length

    def read_and_aggregate_data(self):
        print("=============================================================================\n\n")
        print("\tTraining and evaluating regression model\n\n")
        print("=============================================================================")

        data_dir = "../data/Subject_25"

        subject_ids = []
        all_data = []

        # 0. Read in files
        for subj_folder in os.listdir(data_dir):
            subj_path = os.path.join(data_dir, subj_folder)

            print(subj_path)

            if os.path.exists(subj_path):
                depthCalib_path = os.path.join(subj_path, "depthCalibration.csv")
                if os.path.exists(depthCalib_path):
                    df_depthEval = pd.read_csv(depthCalib_path, delimiter="\t")
                    # Extract and typecast specific columns
                    starting_columns = ['GT depth', 'World Gaze Direction R X', 'World Gaze Direction R Y',
                                        'World Gaze Direction R Z', 'World Gaze Direction L X',
                                        'World Gaze Direction L Y', 'World Gaze Direction L Z',
                                        'World Gaze Origin R X', 'World Gaze Origin R Z',
                                        'World Gaze Origin L X', 'World Gaze Origin L Z']

                    # self.clean_column_names(df_depthEval)
                    df_depthEval = df_depthEval[starting_columns]
                    for col in starting_columns:
                        df_depthEval[col] = df_depthEval[col].astype(float)

                    df_depthEval = base_cleaning(df_depthEval)
                    df_depthEval['SubjectID'] = subj_folder
                    all_data.append(df_depthEval)

                    # Add the subject ID to the list
                    subject_ids.append(subj_folder)

        print("Finished reading subjects and saved sequences.")
        # Combine all data into a single dataframe
        combined_data = pd.concat(all_data, ignore_index=True)

        return combined_data

    def apply_transformation(self, trainset, validset):
        trainset_transformed = self.transform_and_visualize(trainset)
        validationset_transformed = self.apply_transformations_to_validation(validset)

        return trainset_transformed, validationset_transformed

    def apply_transformation_dataset(self, dataset, isTrain=False):
        if isTrain:
            dataset_transformed = self.transform_and_visualize(dataset)
        else:
            dataset_transformed = self.apply_transformations_to_validation(dataset)

        return dataset_transformed

    def transform_and_visualize(self, data_in):
        best_transformers = {}
        transformed_data = data_in.copy()
        ideal_skew = 0.0
        ideal_kurt = 3.0

        for column in data_in.columns:
            if column == "GT depth" or column == "SubjectID":
                continue

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
                    print(f"Transformation failed for {name} on column {column}: {e}")
                    continue

            best_transformers[column] = (best_transform_name, best_transform)

            # Transform the column in the dataset
            if best_transform:
                transformed_column = best_transform.transform(data_in[[column]])
                transformed_data[column] = transformed_column.squeeze()

        self.best_transformers = best_transformers
        return transformed_data

    def apply_transformations_to_validation(self, validation_data):
        transformed_validation_data = validation_data.copy()

        for column, (name, transformer) in self.best_transformers.items():
            if transformer is not None:
                if column == "GT depth":
                    transformed_validation_data[column] = validation_data[[column]]
                elif column == "SubjectID":
                    continue
                else:
                    # Apply the transformation using the fitted transformer object
                    transformed_column = transformer.transform(validation_data[[column]])
                    transformed_validation_data[column] = transformed_column.squeeze()

        return transformed_validation_data

    def scale_target_dataset(self, data_in, isTrain=False):

        if isTrain:
            print("BHO Scaler: Using minmax scaler")
            self.target_scaler = MinMaxScaler(feature_range=(0, 1000))

            # Extract GT depth before scaling and reshape for scaler compatibility
            gt_depth = data_in['GT depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.fit_transform(gt_depth)
                # Re-attach the excluded columns
            data_in['GT depth'] = gt_depth.ravel()
        else:

            gt_depth = data_in['GT depth'].values.reshape(-1, 1)
            # If a feature scaler is set, fit and transform the training data, and transform the validation data
            if self.target_scaler is not None:
                gt_depth = self.target_scaler.transform(gt_depth)
                # Re-attach the excluded columns
            data_in['GT depth'] = gt_depth.ravel()

        return data_in
