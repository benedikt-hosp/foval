import math
import os

import numpy as np
import pandas as pd

from data.AbstractDatasetClass import AbstractDatasetClass
from data.foval_preprocessor import (
    remove_outliers_in_labels, binData,
    detect_and_remove_outliers_in_features_iqr, clean_data
)


class GIWDataset(AbstractDatasetClass):
    def __init__(self, data_dir, trial_name='T1_indoor_walk'):
        super().__init__(data_dir, sequence_length=10)
        self.dataset_name = "GIW"
        self.isGIW = True
        self.trial_name = trial_name

    def load_data(self):
        print(f"Loading data for trial: {self.trial_name}...")
        all_data = []
        trial_dir = os.path.join(self.data_dir, self.trial_name, 'parsed_to_csv')

        for csv_file in os.listdir(trial_dir):
            csv_path = os.path.join(trial_dir, csv_file)
            if not csv_file.endswith('.csv') or not os.path.exists(csv_path):
                continue

            print(f"Processing file: {csv_file}")
            df = pd.read_csv(csv_path, delimiter="\t")
            df.rename(columns=lambda x: x.strip().replace(' ', '_').title(), inplace=True)

            expected_columns = [
                'Gt_Depth', 'World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y',
                'World_Gaze_Direction_L_Z', 'World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y',
                'World_Gaze_Direction_R_Z', 'Vergence_Angle'
            ]
            df = df[expected_columns]

            # Calculate IPD from vergence angle and ground truth depth
            df['IPD'] = df.apply(self._calculate_ipd, axis=1)

            df = clean_data(df, target_column_name=self.target_column_name, multiplication=self.multiplicator)
            df = remove_outliers_in_labels(df, window_size=5, threshold=10, target_column_name=self.target_column_name)
            df = detect_and_remove_outliers_in_features_iqr(df)
            df = binData(df, False)

            subject_id = csv_file.split('_')[0] + '_' + csv_file.split('_')[1]
            df['SubjectID'] = subject_id
            all_data.append(df)

        if all_data:
            self.input_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully loaded {len(all_data)} files with {len(self.input_data)} rows.")
        else:
            print(f"No CSV files found in {trial_dir}.")

        self.subject_list = self.input_data['SubjectID'].unique()
        return self.input_data

    @staticmethod
    def _calculate_ipd(row):
        vergence_angle = pd.to_numeric(row['Vergence_Angle'], errors='coerce')
        focused_depth = pd.to_numeric(row['Gt_Depth'], errors='coerce')

        if pd.isna(vergence_angle) or pd.isna(focused_depth):
            return np.nan

        try:
            ipd = 2 * focused_depth * math.tan(math.radians(vergence_angle) / 2)
        except ValueError:
            return np.nan

        return ipd * 10
