"""
Export cleaned datasets as Parquet files for reproducibility.
Exports data AFTER load_data() (cleaning, outlier removal, binning)
but BEFORE create_features() — so feature engineering and normalization
happen per fold during training (no data leakage).

Usage:
    python export_preprocessed.py
"""
import os
from data.giw_dataset import GIWDataset
from data.robustVision_dataset import RobustVisionDataset
from data.TuftsDataset import TuftsDataset

OUTPUT_DIR = "data/preprocessed"

DATASETS = {
    "giw": lambda: GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making"),
    "robustvision": lambda: RobustVisionDataset(data_dir="data/input/robustvision/"),
    "tufts": lambda: TuftsDataset(data_dir="data/input/tufts/", test_split_size=10),
}


def export_dataset(name, dataset):
    dataset.load_data()
    # Do NOT call create_features() — that must happen per fold to avoid leakage

    data = dataset.input_data
    output_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    data.to_parquet(output_path, index=False, engine="pyarrow")

    n_subjects = len(data['SubjectID'].unique())
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  {name}: {len(data)} rows, {n_subjects} subjects -> {output_path} ({size_kb:.0f} KB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Exporting cleaned datasets (before feature engineering)...\n")
    for name, create_fn in DATASETS.items():
        print(f"Processing {name}...")
        dataset = create_fn()
        export_dataset(name, dataset)
        print()

    print(f"Done. All files saved to {OUTPUT_DIR}/")
    print("Upload this folder to OSF.io for reproducibility.")


if __name__ == "__main__":
    main()
