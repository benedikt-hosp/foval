import math
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from scipy import stats
from sklearn.utils import resample


# original 40
selected_features = [
    'SubjectID',
    'GT depth',
    # INPUT FEATURE
    'World Gaze Direction R X',
    'World Gaze Direction R Y',
    'World Gaze Direction R Z',
    'World Gaze Direction L X',
    'World Gaze Direction L Y',
    'World Gaze Direction L Z',
    'World Gaze Origin R X',
    'World Gaze Origin R Z',
    'World Gaze Origin L X',
    'World Gaze Origin L Z',
    # basic computations
    'Vergence_Angle',
    'Normalized_Vergence_Angle',
    'Vergence_Depth',
    'Cosine_Angles',
    'Delta_Gaze_X',
    'Delta_Gaze_Y',
    'Delta_Gaze_Z',
    'Gaze_Vector_Angle',  # EVA?
    'Gaze_Point_Euclidean_Distance',
    'Directional_Magnitude_R',
    'Directional_Magnitude_L',

    # Ratios and differences
    'Directional_Magnitude_Ratio',
    'Gaze_Direction_X_Ratio',
    'Gaze_Direction_Y_Ratio',
    'Gaze_Direction_Z_Ratio',
    'Ratio_Delta_Gaze_XY',
    'Velocity_X',
    'Acceleration_X',
    'Gaze_Direction_Angle',
    'Relative_Change_Vergence_Angle',

    # Double
    'Angular_Difference_Gaze_Directions',
    'Ratio_Directional_Magnitude',
    'Gaze_Point_Distance',
    'Gaze_Point_Depth_Difference',
    'Ratio_World_Gaze_Direction_X',
    'Ratio_World_Gaze_Direction_Y',
    'Ratio_World_Gaze_Direction_Z',
    'Velocity_Gaze_Direction_R_X',
    'Acceleration_Gaze_Direction_R_X',
    'Angular_Difference_X',
]

# Try 1: no repetition = 29 features
selected_features_no_rep = [
    'SubjectID',
    'GT depth',
    # INPUT FEATURE
    'World Gaze Direction R X',
    'World Gaze Direction R Y',
    'World Gaze Direction R Z',
    'World Gaze Direction L X',
    'World Gaze Direction L Y',
    'World Gaze Direction L Z',
    'World Gaze Origin R X',
    'World Gaze Origin R Z',
    'World Gaze Origin L X',
    'World Gaze Origin L Z',

    # basic computations
    'Normalized_Vergence_Angle',
    'Vergence_Depth',
    'Delta_Gaze_X',
    'Delta_Gaze_Y',
    'Delta_Gaze_Z',
    'Directional_Magnitude_R',
    'Directional_Magnitude_L',

    # Ratios and differences
    'Directional_Magnitude_Ratio',
    'Gaze_Direction_X_Ratio',
    'Gaze_Direction_Y_Ratio',
    'Gaze_Direction_Z_Ratio',
    'Velocity_X',
    'Acceleration_X',
    'Gaze_Direction_Angle',  #

    # Double
    'Angular_Difference_Gaze_Directions',
    'Gaze_Point_Distance',
    'Ratio_World_Gaze_Direction_X',
    'Ratio_World_Gaze_Direction_Y',
    'Ratio_World_Gaze_Direction_Z',
]

# web
selected_features2 = [
    # Features with Similar Names or Directly Related:
    'Vergence_Angle',
    'Vergence_Depth',  # Normalized_Depth
    'Directional_Magnitude_R',
    'Directional_Magnitude_L',
    'Delta_Gaze_X',
    'Delta_Gaze_Y',
    'Delta_Gaze_Z',
    # Features Computed in a Similar Manner:
    'Gaze_Point_Depth_Difference',
    # Delta_Gaze_Z: Both measure differences along the Z-axis but derived from different bases (Gaze Direction vs. Gaze Point Depth).

    'Ratio_Directional_Magnitude',
    'Ratio_Delta_Gaze_XY',
    'Ratio_World_Gaze_Direction_X',
    'Ratio_World_Gaze_Direction_Y',
    'Ratio_World_Gaze_Direction_Z'
    # : These are all ratio-based features comparing aspects of right and left eye measurements.

    'Velocity_Gaze_Direction_R_X',
    'Acceleration_Gaze_Direction_R_X',
    # (also considering similar calculations for Y and Z): These are time-derivative features showing the velocity and acceleration of gaze direction changes.

    # Features that Reflect Similar Concepts Through Different Computations:
    'Cosine_Angles',
    'Gaze_Vector_Angle',
    # : Both relate to angles associated with gaze direction, with the former using the cosine of the vergence angle and the latter derived from the angle between the gaze vectors of the right and left eyes.
    'Normalized_Vergence_Angle',
    'Vergence_Angle',
    # : Both are about the angle of vergence, but the former is a normalized version to scale between -1 and 1.
    'Directional_Magnitude_Ratio',
    'Gaze_Direction_X_Ratio',
    'Gaze_Direction_Y_Ratio',
    'Gaze_Direction_Z_Ratio',
    # : These ratios compare the magnitudes or components of gaze direction vectors between the right and left eyes in different ways but fundamentally express relationships between the two eyes' gaze directions.
    'Angular_Difference_Gaze_Directions',
    'Gaze_Direction_Angle'
    # Both describe angular differences between gaze directions, with slightly different computation methods (direct calculation vs. using dot products and magnitudes).

]


def detect_and_remove_outliers_in_features(df):

    # Assuming df is your DataFrame
    features = df.drop(columns=['GT depth'])

    # Apply Standard Scaler
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)

    # Compute Z-scores on the scaled data
    z_scores = np.abs(stats.zscore(scaled_features))

    # Define a threshold
    threshold = 3  # You can adjust this value

    # Filter entries with Z-score less than the threshold
    filtered_entries = (z_scores < threshold).all(axis=1)

    # Create a new DataFrame with the filtered entries
    new_df = df[filtered_entries]

    return new_df


def detect_and_remove_outliers_in_features_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1


    mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.0 * IQR))

    # Filter out the rows with outliers
    new_df = df[~mask.any(axis=1)]
    return new_df


def detect_and_remove_outliers(df, window_size, threshold):
    # Check if 'GT depth' column exists
    if 'GT depth' not in df.columns:
        raise ValueError("Column 'GT depth' not found in the DataFrame")

    # Iterate over the DataFrame
    outlier_indices = []
    for i in range(len(df)):
        # Define the window range
        start = max(i - window_size // 2, 0)
        end = min(i + window_size // 2 + 1, len(df))
        window = df['GT depth'].iloc[start:end]

        # Calculate the median of the window
        mean = np.mean(window)
        # median = np.nanmedian(window)

        # # Check if the current value is an outlier
        if abs(df['GT depth'].iloc[i] - mean) > threshold:
            outlier_indices.append(i)

        # Check if the current value is an outlier
        # if abs(df['GT depth'].iloc[i] - median) > threshold:
        #     outlier_indices.append(i)

    # Check if the outlier indices are in the DataFrame index
    outlier_indices = [idx for idx in outlier_indices if idx in df.index]
    # Now drop the outliers safely
    df_cleaned = df.drop(outlier_indices)
    print(f"Removed {len(outlier_indices)} outlier from data set.")

    return df_cleaned


def split_data_by_subjects(scaled_data, train_size=0.9):
    # Get unique subject IDs
    unique_subjects = scaled_data['SubjectID'].unique()

    # Split subject IDs into training and validation sets
    train_subjects, validation_subjects = train_test_split(unique_subjects, train_size=train_size, shuffle=True)

    # Filter data by subject IDs for training and validation sets
    train_data = scaled_data[scaled_data['SubjectID'].isin(train_subjects)]
    validation_data = scaled_data[scaled_data['SubjectID'].isin(validation_subjects)]

    # now drop the columns as we do not need them anymore
    # Drop the 'SubjectID' column from both datasets
    # train_data = train_data.drop(columns=['SubjectID'])
    # validation_data = validation_data.drop(columns=['SubjectID'])

    return train_data, validation_data


def getIPD(row):
    posR = [row['World Gaze Origin R X'], 0.0, row['World Gaze Origin R Z']]
    posL = [row['World Gaze Origin L X'], 0.0, row['World Gaze Origin L Z']]
    deltaX = posR[0] - posL[0]
    deltaY = posR[1] - posL[1]
    deltaZ = posR[2] - posL[2]
    IPD = math.sqrt((deltaX ** 2) + (deltaY ** 2) + (deltaZ ** 2))
    return IPD


def getAngle(row):
    vecR = [row['World Gaze Direction R X'], row['World Gaze Direction R Y'], row['World Gaze Direction R Z']]
    vecL = [row['World Gaze Direction L X'], row['World Gaze Direction L Y'], row['World Gaze Direction L Z']]
    vecR_n = np.linalg.norm(vecR)
    vecL_n = np.linalg.norm(vecL)
    angle = np.arccos(np.dot(vecR, vecL) / (vecR_n * vecL_n))
    return np.degrees(angle)


def global_normalization(data):
    features = data.drop(columns=['SubjectID', 'GT depth'])
    # scaler = StandardScaler()  # Global scaler
    scaler = RobustScaler()  # Global scaler

    normalized_features = scaler.fit_transform(features)
    data_normalized = pd.DataFrame(normalized_features, columns=features.columns)
    data_normalized['SubjectID'] = data['SubjectID'].values
    data_normalized['GT depth'] = data['GT depth'].values
    return data_normalized


def getEyeVergenceAngle(row):
    vergenceAngle = getAngle(row)
    if vergenceAngle != 0:
        ipd = getIPD(row)
        # Added a check to prevent division by zero if vergenceAngle is extremely small
        depth = ipd / (2 * math.tan(math.radians(vergenceAngle) / 2)) if math.radians(vergenceAngle) != 0 else 0
    else:
        depth = 0.0
    # Convert depth to millimeters by multiplying by 1000 (from meters to mm)
    depth_fin = abs(depth) * 100  # Convert from meters to millimeters
    return vergenceAngle, depth_fin


def normalize_subject_data(data, scaler):
    # Normalize data for a single subject using the provided scaler
    # Assuming 'SubjectID' is not to be normalized
    features = data.drop(columns=['SubjectID', 'GT depth'])
    normalized_features = scaler.fit_transform(features)
    data_normalized = pd.DataFrame(normalized_features, columns=features.columns)
    data_normalized['SubjectID'] = data['SubjectID'].values
    data_normalized['GT depth'] = data['GT depth'].values
    return data_normalized


def subject_wise_normalization(data, unique_subjects, scaler):
    normalized_data_list = []
    for subject in unique_subjects:
        subject_data = data[data['SubjectID'] == subject]
        subject_data_normalized = normalize_subject_data(subject_data, scaler)
        normalized_data_list.append(subject_data_normalized)
    return pd.concat(normalized_data_list, ignore_index=True)


def subjective_normalization_dataset(data_set_in):
    # Apply global normalization first
    data_set_in = global_normalization(data_set_in)

    # Then proceed with the existing subject-wise normalization
    unique_subjects = data_set_in['SubjectID'].unique()

    # Choose your scaler for subject-wise normalization
    subject_scaler = RobustScaler()  # or any other scaler
    # subject_scaler = QuantileTransformer(output_distribution='normal')  # or any other scaler

    # Apply subject-wise normalization
    dataset_in_normalized = subject_wise_normalization(data_set_in, unique_subjects, subject_scaler)

    return dataset_in_normalized


def subjective_normalization(train_data, validation_data):
    # Apply global normalization first
    train_data = global_normalization(train_data)
    validation_data = global_normalization(validation_data)

    # Then proceed with the existing subject-wise normalization
    unique_train_subjects = train_data['SubjectID'].unique()
    unique_validation_subjects = validation_data['SubjectID'].unique()

    # Choose your scaler for subject-wise normalization
    subject_scaler = RobustScaler()  # or any other scaler
    # subject_scaler = QuantileTransformer(output_distribution='normal')  # or any other scaler

    # Apply subject-wise normalization
    training_set_normalized = subject_wise_normalization(train_data, unique_train_subjects, subject_scaler)
    validation_set_normalized = subject_wise_normalization(validation_data, unique_validation_subjects,
                                                                subject_scaler)

    return training_set_normalized, validation_set_normalized


def createFeatures(data_in):

    # data_in['Vergence_Angle'], data_in['Vergence_Depth'] = zip(*data_in.apply(self.getEyeVergenceAngle, axis=1))

    data_in['Vergence_Angle'], data_in['Vergence_Depth'] = zip(*data_in.apply(getEyeVergenceAngle, axis=1))

    # 1. Depth Normalization
    max_depth = data_in['Vergence_Depth'].max()
    min_depth = data_in['Vergence_Depth'].min()

    # max_depth = 1000.0
    # min_depth = 0.0
    data_in['Normalized_Depth'] = (data_in['Vergence_Depth'] - min_depth) / (max_depth - min_depth)

    # 2. Directional Magnitude for Right and Left
    data_in['Directional_Magnitude_R'] = np.linalg.norm(
        data_in[['World Gaze Direction R X', 'World Gaze Direction R Y', 'World Gaze Direction R Z']].values,
        axis=1)
    data_in['Directional_Magnitude_L'] = np.linalg.norm(
        data_in[['World Gaze Direction L X', 'World Gaze Direction L Y', 'World Gaze Direction L Z']].values,
        axis=1)

    # 3. Gaze Direction Cosine Angles (already computed as Vergence_Angle)
    data_in['Cosine_Angles'] = np.cos(np.radians(data_in['Vergence_Angle']))

    # 4. Gaze Point Distance
    data_in['Gaze_Point_Distance'] = np.sqrt(
        (data_in['World Gaze Direction R X'] - data_in['World Gaze Direction L X']) ** 2 +
        (data_in['World Gaze Direction R Y'] - data_in['World Gaze Direction L Y']) ** 2)

    # 5. Normalized Vergence Angle
    max_angle = data_in['Vergence_Angle'].max()
    min_angle = data_in['Vergence_Angle'].min()
    data_in['Normalized_Vergence_Angle'] = 2 * (
            (data_in['Vergence_Angle'] - min_angle) / (max_angle - min_angle)) - 1

    # 6. Difference in World Gaze Direction
    data_in['Delta_Gaze_X'] = data_in['World Gaze Direction R X'] - data_in['World Gaze Direction L X']
    data_in['Delta_Gaze_Y'] = data_in['World Gaze Direction R Y'] - data_in['World Gaze Direction L Y']
    data_in['Delta_Gaze_Z'] = data_in['World Gaze Direction R Z'] - data_in['World Gaze Direction L Z']

    # 1. Angle between Gaze Vectors
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(np.clip(dot_product / magnitude, -1.0, 1.0))

    gaze_r = data_in[['World Gaze Direction R X', 'World Gaze Direction R Y', 'World Gaze Direction R Z']].values
    gaze_l = data_in[['World Gaze Direction L X', 'World Gaze Direction L Y', 'World Gaze Direction L Z']].values
    data_in['Gaze_Vector_Angle'] = [angle_between_vectors(gaze_r[i], gaze_l[i]) for i in range(len(gaze_r))]

    # 2. Gaze Point Depth
    data_in['Gaze_Point_Depth_Difference'] = data_in['World Gaze Direction R Z'] - data_in[
        'World Gaze Direction L Z']

    # 3. Relative Changes
    # data_in['Relative_Change_Normalized_Depth'] = data_in['Normalized_Depth'].diff().fillna(0)
    data_in['Relative_Change_Vergence_Angle'] = data_in['Vergence_Angle'].diff().fillna(0)

    # 4. Ratios
    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_Delta_Gaze_XY'] = data_in['Delta_Gaze_X'] / data_in['Delta_Gaze_Y']

    # data_in['Relative_Change_Normalized_Depth'] = data_in['Normalized_Depth'].diff().fillna(0)
    data_in['Relative_Change_Vergence_Angle'] = data_in['Vergence_Angle'].diff().fillna(0)

    data_in['Ratio_Directional_Magnitude'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Ratio_World_Gaze_Direction_X'] = data_in['World Gaze Direction R X'] / data_in[
        'World Gaze Direction L X']
    data_in['Ratio_World_Gaze_Direction_Y'] = data_in['World Gaze Direction R Y'] / data_in[
        'World Gaze Direction L Y']
    data_in['Ratio_World_Gaze_Direction_Z'] = data_in['World Gaze Direction R Z'] / data_in[
        'World Gaze Direction L Z']

    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    # data_in['Velocity_Gaze_Direction_R_X'] = data_in['World Gaze Direction R X'].diff().fillna(0)
    # data_in['Acceleration_Gaze_Direction_R_X'] = data_in['Velocity_Gaze_Direction_R_X'].diff().fillna(0)
    # Repeat for Y and Z

    data_in['Angular_Difference_Gaze_Directions'] = data_in['Gaze_Vector_Angle']  # Already computed above

    data_in['Interaction_Normalized_Depth_Vergence_Angle'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Assuming a rolling window of size 5
    data_in['Rolling_Mean_Normalized_Depth'] = data_in['Normalized_Depth'].rolling(window=5).mean().fillna(0)

    data_in['Lag_1_Normalized_Depth'] = data_in['Normalized_Depth'].shift(1).fillna(0)

    # Relative Changes
    data_in['Diff_Normalized_Depth'] = data_in['Normalized_Depth'].diff()

    # Ratios
    data_in['Directional_Magnitude_Ratio'] = data_in['Directional_Magnitude_R'] / data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_X_Ratio'] = data_in['World Gaze Direction R X'] / data_in['World Gaze Direction L X']
    data_in['Gaze_Direction_Y_Ratio'] = data_in['World Gaze Direction R Y'] / data_in['World Gaze Direction L Y']
    data_in['Gaze_Direction_Z_Ratio'] = data_in['World Gaze Direction R Z'] / data_in['World Gaze Direction L Z']

    # Velocity and Acceleration
    # data_in['Velocity_X'] = data_in['World Gaze Direction R X'].diff()
    # data_in['Acceleration_X'] = data_in['Velocity_X'].diff()

    # Angular Differences
    data_in['Angular_Difference_X'] = data_in['World Gaze Direction R X'] - data_in['World Gaze Direction L X']

    # Higher Order Interactions (example)
    data_in['Depth_Angle_Interaction'] = data_in['Normalized_Depth'] * data_in['Vergence_Angle']

    # Distance between Gaze Points
    data_in['Gaze_Point_Euclidean_Distance'] = np.sqrt(
        (data_in['World Gaze Direction R X'] - data_in['World Gaze Direction L X']) ** 2 +
        (data_in['World Gaze Direction R Y'] - data_in['World Gaze Direction L Y']) ** 2
    )

    # Angle between Gaze Directions (using dot product)
    dot_product = (
            data_in['World Gaze Direction R X'] * data_in['World Gaze Direction L X'] +
            data_in['World Gaze Direction R Y'] * data_in['World Gaze Direction L Y'] +
            data_in['World Gaze Direction R Z'] * data_in['World Gaze Direction L Z']
    )
    magnitude_R = data_in['Directional_Magnitude_R']
    magnitude_L = data_in['Directional_Magnitude_L']
    data_in['Gaze_Direction_Angle'] = np.arccos(dot_product / (magnitude_R * magnitude_L))

    # Drop NaN values created by diff() function
    data_in = data_in.dropna()

    data_in = data_in.replace([np.inf, -np.inf], np.nan)
    data_in = data_in.dropna()
    # Define excluded features
    excluded_features = ['World Gaze Origin R X', 'World Gaze Origin R Z', 'World Gaze Origin L X',
                         'World Gaze Origin L Z']

    # Remove excluded features
    data_in = data_in.drop(columns=excluded_features)

    return data_in


def separate_features_and_targets(sequences):
    features = [seq[0] for seq in sequences]
    targets = [seq[1] for seq in sequences]
    return features, targets


def augmentTrainingData(training_set=None):

    training_set = training_set.replace([np.inf, -np.inf], np.nan)
    training_set = training_set.dropna()

    # Separate the columns to be excluded from augmentation
    y_train = training_set['GT depth']
    subject_id = training_set['SubjectID']

    # Drop the columns 'GT depth' and 'SubjectID' from the training set before augmentation
    X_train = training_set.drop(['GT depth', 'SubjectID'], axis=1)
    print("Regression on ", X_train.columns)
    augmentData = True
    if augmentData:
        # ============== DATA AUGMENTATION =============
        X_train_augmented = X_train.copy()  # Ensure a separate copy for augmentation
        X_train_augmented = add_gaussian_noise(X_train_augmented)
        # If you decide to use jitter in the future, uncomment the following line
        # X_train_augmented = jitter(X_train_augmented)

        # The target values remain the same for augmented data
        y_train_augmented = y_train.copy()

        # Use only augmented data
        X_train_combined = X_train_augmented
        y_train_combined = y_train_augmented

    else:
        # Use the original data as the combined dataset
        X_train_combined = X_train
        y_train_combined = y_train

    # Since you're using augmented data only when augmentData is True, or original otherwise,
    # the subject_id_combined handling can stay outside the if-else block as it applies in both cases
    subject_id_combined = subject_id

    # Convert the combined data back to a DataFrame
    training_set_combined_df = pd.DataFrame(X_train_combined, columns=X_train.columns)
    training_set_combined_df['GT depth'] = y_train_combined
    training_set_combined_df['SubjectID'] = subject_id_combined
    training_set_combined_df.dropna(inplace=True)

    return training_set_combined_df


def augment_data(df, noise_level=0.1, shift_max=5, scaling_factor_range=(0.9, 1.1)):
    augmented_data = df.copy()

    # Noise Injection
    for col in df.columns:
        if col != 'SubjectID' and col != 'GT depth':
            noise = np.random.normal(0, noise_level, size=df[col].shape)
            augmented_data[col] += noise

    # Time Shifting
    shift = random.randint(-shift_max, shift_max)
    augmented_data = augmented_data.shift(shift).fillna(method='bfill')

    # Scaling
    scaling_factor = random.uniform(*scaling_factor_range)
    for col in df.columns:
        if col != 'SubjectID' and col != 'GT depth':
            augmented_data[col] *= scaling_factor

    # Example of checking for NaN values before processing
    if augmented_data.isna().any().any():
        # Handle NaN values
        augmented_data = augmented_data.fillna(method='ffill')  # Forward fill as an example

    return augmented_data


def binData(df):

    # Step 1: Bin the target variable
    num_bins = 60  # You can adjust this number
    df['GT_depth_bin'] = pd.cut(df['GT depth'], bins=num_bins, labels=False)

    # Step 2: Calculate mean count per bin
    bin_counts = df['GT_depth_bin'].value_counts()
    mean_count = bin_counts.mean()

    # Step 3: Resample each bin
    resampled_data = []
    for bin in range(num_bins):
        bin_data = df[df['GT_depth_bin'] == bin]
        bin_count = bin_data.shape[0]

        if bin_count == 0:
            continue  # Skip empty bins

        if bin_count < mean_count:
            # Oversample if count is less than mean
            bin_data_resampled = resample(bin_data, replace=True, n_samples=int(mean_count), random_state=123)
        elif bin_count > mean_count:
            # Undersample if count is more than mean
            bin_data_resampled = resample(bin_data, replace=False, n_samples=int(mean_count), random_state=123)
        else:
            # Keep the bin as is if count is equal to mean
            bin_data_resampled = bin_data

        resampled_data.append(bin_data_resampled)

    # Step 4: Combine back into a single DataFrame
    balanced_df = pd.concat(resampled_data)

    # Optionally, drop the 'GT_depth_bin' column if no longer needed
    balanced_df.drop('GT_depth_bin', axis=1, inplace=True)
    return balanced_df

