import os
import pandas as pd

from data.AbstractDatasetClass import AbstractDatasetClass
from data.foval_preprocessor import (
    remove_outliers_in_labels, binData,
    detect_and_remove_outliers_in_features_iqr, clean_data
)


class RobustVisionDataset(AbstractDatasetClass):
    def __init__(self, data_dir):
        super().__init__(data_dir, sequence_length=10)
        self.dataset_name = "robustvision"
        self.isGIW = False

    def load_data(self):
        print("Reading and aggregating data...")
        all_data = []

        for subj_folder in os.listdir(self.data_dir):
            subj_path = os.path.join(self.data_dir, subj_folder)
            depthCalib_path = os.path.join(subj_path, "depthCalibration.csv")

            if not os.path.exists(depthCalib_path):
                continue

            df = pd.read_csv(depthCalib_path, delimiter="\t")
            df.rename(columns=lambda x: x.strip().replace(' ', '_').title(), inplace=True)

            expected_columns = [
                'Gt_Depth', 'World_Gaze_Direction_L_X', 'World_Gaze_Direction_L_Y',
                'World_Gaze_Direction_L_Z', 'World_Gaze_Direction_R_X', 'World_Gaze_Direction_R_Y',
                'World_Gaze_Direction_R_Z', 'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Z',
                'World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Z'
            ]

            df = df[expected_columns].astype(float)
            df = clean_data(df, target_column_name=self.target_column_name, multiplication=self.multiplicator)
            df = remove_outliers_in_labels(df, window_size=5, threshold=10, target_column_name=self.target_column_name)
            df = detect_and_remove_outliers_in_features_iqr(df)
            df = binData(df, False)
            df['SubjectID'] = subj_folder
            all_data.append(df)

        self.input_data = pd.concat(all_data, ignore_index=True)
        self.subject_list = self.input_data['SubjectID'].unique()
        print(f"Finished loading {len(all_data)} subjects with {len(self.input_data)} rows.")
        return self.input_data
