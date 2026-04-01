import os
import pandas as pd

from data.AbstractDatasetClass import AbstractDatasetClass
from data.foval_preprocessor import (
    remove_outliers_in_labels, binData,
    detect_and_remove_outliers_in_features_iqr, clean_data
)


class TuftsDataset(AbstractDatasetClass):
    def __init__(self, data_dir, test_split_size=10):
        super().__init__(data_dir, sequence_length=10)
        self.dataset_name = "TUFTS"
        self.isGIW = False
        self.test_split_size = test_split_size

    def load_data(self):
        all_data = []
        trial_dir = os.path.join(self.data_dir, 'parsed_to_csv')

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
                'World_Gaze_Direction_R_Z', 'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Y',
                'World_Gaze_Origin_R_Z', 'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Y',
                'World_Gaze_Origin_L_Z'
            ]
            df = df[expected_columns]

            subject_id = csv_file.split('_')[0] + '_' + csv_file.split('_')[1]
            print(f"Subject is {subject_id}")

            df = clean_data(df, target_column_name=self.target_column_name, multiplication=self.multiplicator)
            df = remove_outliers_in_labels(df, window_size=5, threshold=10, target_column_name=self.target_column_name)
            df = detect_and_remove_outliers_in_features_iqr(df)
            df = binData(df, False)
            df['SubjectID'] = subject_id
            all_data.append(df)

        if all_data:
            self.input_data = pd.concat(all_data, ignore_index=True)
            print(f"Successfully loaded {len(all_data)} files with {len(self.input_data)} rows.")
        else:
            print(f"No CSV files found in {trial_dir}.")

        self.subject_list = self.input_data['SubjectID'].unique()
        return self.input_data
