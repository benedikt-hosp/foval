import os
import pandas as pd
import numpy as np

from data.AbstractDatasetClass import AbstractDatasetClass
from data.foval_preprocessor import input_features, createFeatures, separate_features_and_targets
from data.utilities import create_lstm_tensors_dataset, create_dataloaders_dataset

# Map dataset name -> isGIW flag (needed for correct vergence calculation)
DATASET_IS_GIW = {
    "giw": True,
    "robustvision": False,
    "tufts": False,
}


class PreprocessedDataset(AbstractDatasetClass):
    """Loads cleaned Parquet files. Feature engineering + normalization happen per fold."""

    def __init__(self, parquet_dir="data/preprocessed", datasets=None):
        super().__init__(data_dir=parquet_dir, sequence_length=10)
        self.dataset_name = "Preprocessed"
        self.parquet_dir = parquet_dir
        self.datasets = datasets  # e.g. ["giw", "robustvision", "tufts"] or None for all
        self._dataset_sources = {}  # SubjectID -> dataset_name mapping

    def load_data(self):
        parquet_files = self._get_parquet_files()
        all_data = []

        for filepath in parquet_files:
            name = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Loading {name} from {filepath}...")
            df = pd.read_parquet(filepath, engine="pyarrow")

            # Track which dataset each subject comes from (for isGIW flag)
            for subj in df['SubjectID'].unique():
                self._dataset_sources[subj] = name

            all_data.append(df)
            print(f"  {name}: {len(df)} rows, {len(df['SubjectID'].unique())} subjects")

        self.input_data = pd.concat(all_data, ignore_index=True)
        self.subject_list = self.input_data['SubjectID'].unique()

        print(f"\nTotal: {len(self.input_data)} rows, {len(self.subject_list)} subjects")

    def create_features(self, data_in, is_train=True):
        """Run feature engineering per source dataset (each needs its own isGIW flag)."""
        results = []

        for source_name, group in self._group_by_source(data_in):
            is_giw = DATASET_IS_GIW.get(source_name, False)
            result = createFeatures(group, isGIW=is_giw, is_train=is_train, dataset=self)
            results.append(result)

        return pd.concat(results, ignore_index=True)

    def _group_by_source(self, data):
        """Group data rows by their source dataset, dropping NaN-only columns from concat."""
        data = data.copy()
        data['_source'] = data['SubjectID'].map(self._dataset_sources)

        for source_name, group in data.groupby('_source'):
            group = group.drop(columns=['_source'])
            # Drop columns that are all NaN (artifacts from concat of different schemas)
            group = group.dropna(axis=1, how='all')
            yield source_name, group

    def _get_parquet_files(self):
        if self.datasets:
            files = [os.path.join(self.parquet_dir, f"{name}.parquet") for name in self.datasets]
            for f in files:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Parquet file not found: {f}")
            return files

        files = sorted([
            os.path.join(self.parquet_dir, f)
            for f in os.listdir(self.parquet_dir)
            if f.endswith('.parquet')
        ])
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {self.parquet_dir}")
        return files
