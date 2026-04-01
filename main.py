import random
import os
import shutil

import torch
import numpy as np
import pandas as pd
import scipy.io
from sklearn.cluster import KMeans

from data.MixedDatasetClass import MixedDatasetClass
from data.PreprocessedDataset import PreprocessedDataset
from data.SpecificMixDatasetClass import SpecificMixDatasetClass
from data.TuftsDataset import TuftsDataset
from data.foval_preprocessor import input_features
from data.giw_dataset import GIWDataset
from data.robustVision_dataset import RobustVisionDataset
from training.foval_trainer import FOVALTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def seed_everything(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_model_directory(model_save_dir="models"):
    os.makedirs(model_save_dir, exist_ok=True)
    return model_save_dir


# --- Data parsing utilities ---

def process_mat_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

    for mat_file in mat_files:
        print(f"Processing {mat_file}:")
        mat_data = scipy.io.loadmat(os.path.join(input_folder, mat_file))
        ProcessData = mat_data.get('ProcessData', {})

        if ProcessData is None or not ProcessData:
            print(f"Skipping {mat_file} due to missing 'ProcessData'.")
            continue

        try:
            T = ProcessData['T'].flatten()
            T_list = T[0].flatten().tolist()
            SceneDepth = ProcessData['SceneDepth'].flatten()
            Sc_list = (SceneDepth[0].flatten() / 1000).tolist()

            Gaze_Vector_L = ProcessData['ETG'][0][0]['EIHvector_0'][0][0]
            Gaze_Vector_R = ProcessData['ETG'][0][0]['EIHvector_1'][0][0]
            Vergence_Angle = ProcessData['ETG'][0][0]['Vergence'][0][0].flatten()
            Vergence_Angle_list = pd.to_numeric(Vergence_Angle, errors='coerce').tolist()

            data_dict = {
                'Time': T_list,
                'Gt_Depth': Sc_list,
                'World_Gaze_Direction_L_X': [Gaze_Vector_L[i, 0] if i < len(Gaze_Vector_L) else np.nan for i in range(len(T_list))],
                'World_Gaze_Direction_L_Y': [Gaze_Vector_L[i, 1] if i < len(Gaze_Vector_L) else np.nan for i in range(len(T_list))],
                'World_Gaze_Direction_L_Z': [Gaze_Vector_L[i, 2] if i < len(Gaze_Vector_L) else np.nan for i in range(len(T_list))],
                'World_Gaze_Direction_R_X': [Gaze_Vector_R[i, 0] if i < len(Gaze_Vector_R) else np.nan for i in range(len(T_list))],
                'World_Gaze_Direction_R_Y': [Gaze_Vector_R[i, 1] if i < len(Gaze_Vector_R) else np.nan for i in range(len(T_list))],
                'World_Gaze_Direction_R_Z': [Gaze_Vector_R[i, 2] if i < len(Gaze_Vector_R) else np.nan for i in range(len(T_list))],
                'Vergence_Angle': Vergence_Angle_list[:len(T_list)],
            }

            df = pd.DataFrame(data_dict)
            output_csv_file = os.path.join(output_folder, f"{mat_file.replace('.mat', '.csv')}")
            df.to_csv(output_csv_file, index=False, sep='\t')
            print(f"Processed {mat_file} and saved to {output_csv_file}")

        except (KeyError, IndexError) as e:
            print(f"Error processing {mat_file}: {e}")
            continue


def delete_folder_contents(folder_path):
    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def clear_output_folder(output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)


def parseGIWRawData(folder):
    trials = {
        "Trial 1": './data/input/gaze_in_wild/T1_indoor_walk/',
        "Trial 2": './data/input/gaze_in_wild/T2_ball_catch/',
        "Trial 3": './data/input/gaze_in_wild/T3_visual_search/',
        "Trial 4": './data/input/gaze_in_wild/T4_tea_making/'
    }
    for trial_name, trial_path in trials.items():
        print(f"\n\nWorking on {trial_name}")
        input_folder = trial_path + folder
        output_folder = trial_path + 'parsed_to_csv'
        delete_folder_contents(output_folder)
        process_mat_files(input_folder, output_folder)


def process_tufts_data(input_file, output_folder):
    df = pd.read_csv(input_file)
    df_transformed = pd.DataFrame({
        'Gt_Depth': df['distance'],
        'World_Gaze_Direction_L_X': df['gaze_normal0_x'],
        'World_Gaze_Direction_L_Y': df['gaze_normal0_y'],
        'World_Gaze_Direction_L_Z': df['gaze_normal0_z'],
        'World_Gaze_Direction_R_X': df['gaze_normal1_x'],
        'World_Gaze_Direction_R_Y': df['gaze_normal1_y'],
        'World_Gaze_Direction_R_Z': df['gaze_normal1_z'],
        'World_Gaze_Origin_L_X': df['eye_center0_3d_x'],
        'World_Gaze_Origin_L_Y': df['eye_center0_3d_y'],
        'World_Gaze_Origin_L_Z': df['eye_center0_3d_z'],
        'World_Gaze_Origin_R_X': df['eye_center1_3d_x'],
        'World_Gaze_Origin_R_Y': df['eye_center1_3d_y'],
        'World_Gaze_Origin_R_Z': df['eye_center1_3d_z']
    })

    num_subjects = 8
    split_data = np.array_split(df_transformed, num_subjects)

    os.makedirs(output_folder, exist_ok=True)
    delete_folder_contents(output_folder)

    for i, subject_data in enumerate(split_data):
        output_file = os.path.join(output_folder, f'subject_{i + 1}.csv')
        subject_data.to_csv(output_file, index=False, sep="\t")
        print(f"Saved data for Subject {i + 1} to {output_file}")


def process_gaze360_mat_data_fixed(mat_data, output_folder):
    mat_data = scipy.io.loadmat(mat_data)

    target_pos3d = mat_data['target_pos3d']
    person_eyes3d = mat_data['person_eyes3d']

    valid_scene_depths = np.sqrt(
        (target_pos3d[:, 0] - person_eyes3d[:, 0]) ** 2 +
        (target_pos3d[:, 1] - person_eyes3d[:, 1]) ** 2 +
        (target_pos3d[:, 2] - person_eyes3d[:, 2]) ** 2
    )

    data_dict = {
        'Time': mat_data['ts'].flatten(),
        'Gt_Depth': valid_scene_depths,
        'World_Gaze_Direction_L_X': mat_data['gaze_dir'][:, 0],
        'World_Gaze_Direction_L_Y': mat_data['gaze_dir'][:, 1],
        'World_Gaze_Direction_L_Z': mat_data['gaze_dir'][:, 2],
        'World_Gaze_Direction_R_X': mat_data['gaze_dir'][:, 0],
        'World_Gaze_Direction_R_Y': mat_data['gaze_dir'][:, 1],
        'World_Gaze_Direction_R_Z': mat_data['gaze_dir'][:, 2],
        'World_Gaze_Origin_R_X': person_eyes3d[:, 0],
        'World_Gaze_Origin_R_Y': person_eyes3d[:, 1],
        'World_Gaze_Origin_R_Z': person_eyes3d[:, 2],
        'World_Gaze_Origin_L_X': person_eyes3d[:, 0],
        'World_Gaze_Origin_L_Y': person_eyes3d[:, 1],
        'World_Gaze_Origin_L_Z': person_eyes3d[:, 2],
    }

    df = pd.DataFrame(data_dict)
    df['person_identity'] = mat_data['person_identity'].flatten()

    gaze_origins = df[['World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Z',
                        'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Z']].values

    kmeans = KMeans(n_clusters=238, random_state=42)
    df['clustered_participant'] = kmeans.fit_predict(gaze_origins)

    clear_output_folder(output_folder)

    for participant in np.unique(df['clustered_participant']):
        participant_data = df[df['clustered_participant'] == participant]
        if participant_data[participant_data['Gt_Depth'] < 3].shape[0] > 0:
            output_file = f"{output_folder}participant_{int(participant)}.csv"
            participant_data.to_csv(output_file, index=False, sep="\t")
            print(f"Saved data for clustered participant {int(participant)} to {output_file}")
        else:
            print(f"Skipped participant_{int(participant)}: No depths under 3 meters.")


# --- Experiment functions ---

def final_superset_check(combined_dataset):
    data = combined_dataset.input_data.fillna(0)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    combined_dataset.input_data = data
    print(f"Combined Data after cleanup: {len(data)} rows")
    return combined_dataset


def a_single_sets(model_type):
    dataset = GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making")
    dataset.load_data()

    foval_trainer = FOVALTrainer(
        config_path="models/config/foval.json", dataset=dataset, device=device,
        feature_names=input_features, save_intermediates_every_epoch=False, model_type=model_type)
    foval_trainer.setup()
    mean_mae = foval_trainer.cross_validate(num_epochs=n_epochs)
    print(f"Model Average Mean Absolute Error: {mean_mae}")


def b_loocv_all():
    data_dirs = {
        'robustvision': 'data/input/robustvision/',
        'giw': 'data/input/gaze_in_wild/',
        'tufts': 'data/input/tufts/'
    }

    combined_dataset = MixedDatasetClass(data_dirs=data_dirs)
    combined_dataset.load_data()
    combined_dataset = final_superset_check(combined_dataset)

    foval_trainer = FOVALTrainer(
        config_path="models/config/foval.json", dataset=combined_dataset, device=device,
        feature_names=list(combined_dataset.input_data.columns),
        save_intermediates_every_epoch=False, model_type="LSTM")
    foval_trainer.setup()
    mean_mae = foval_trainer.cross_validate(num_epochs=n_epochs)
    print(f"Model Average Mean Absolute Error: {mean_mae}")


def c_cross_datasets():
    all_sets = {
        'robustvision': 'data/input/robustvision/',
        'giw': 'data/input/gaze_in_wild/',
        'tufts': 'data/input/tufts/'
    }

    for train_set, train_path in all_sets.items():
        if train_set == 'robustvision':
            continue
        for val_set, val_path in all_sets.items():
            if train_set == val_set:
                continue

            print(f"\nTraining on {train_set} and validating on {val_set}")
            specific_dataset = SpecificMixDatasetClass({train_set: train_path}, {val_set: val_path}, None, 10)
            specific_dataset.load_data()
            specific_dataset = final_superset_check(specific_dataset)

            foval_trainer = FOVALTrainer(
                config_path="models/config/foval.json", dataset=specific_dataset, device=device,
                feature_names=list(specific_dataset.train_data.columns),
                save_intermediates_every_epoch=False, model_type="LSTM")
            foval_trainer.setup()

            mean_mae = foval_trainer.cross_validate_with_specific_datasets(num_epochs=n_epochs)
            print(f"Cross-validation Mean MAE for {train_set} -> {val_set}: {mean_mae}")


def d_pretraining():
    all_sets = {
        'robustvision': 'data/input/robustvision/',
        'giw': 'data/input/gaze_in_wild/',
        'tufts': 'data/input/tufts/'
    }

    for val_set, val_path in all_sets.items():
        print(f"\nStarting Validation on val_set: {val_set}")
        train_dirs = {k: v for k, v in all_sets.items() if k != val_set}

        specific_dataset = SpecificMixDatasetClass(train_dirs, {val_set: val_path}, None, 10)
        specific_dataset.load_data()

        foval_trainer = FOVALTrainer(
            config_path="models/config/foval.json", dataset=specific_dataset, device=device,
            feature_names=list(specific_dataset.train_data.columns),
            save_intermediates_every_epoch=False, model_type="LSTM")
        foval_trainer.setup()

        mean_mae = foval_trainer.cross_validate_with_specific_datasets(num_epochs=n_epochs)
        print(f"Cross-validation Mean MAE: {mean_mae}")


def e_preprocessed_loocv(model_type="LSTM"):
    """Train using preprocessed Parquet files (no raw data needed)."""
    dataset = PreprocessedDataset(parquet_dir="data/preprocessed", datasets=["giw", "robustvision", "tufts"])
    dataset.load_data()
    dataset = final_superset_check(dataset)

    foval_trainer = FOVALTrainer(
        config_path="models/config/foval.json", dataset=dataset, device=device,
        feature_names=list(dataset.input_data.columns),
        save_intermediates_every_epoch=False, model_type=model_type)
    foval_trainer.setup()
    mean_mae = foval_trainer.cross_validate(num_epochs=n_epochs)
    print(f"Model Average Mean Absolute Error: {mean_mae}")


if __name__ == "__main__":
    seed_everything(seed=42)
    device = setup_device()
    n_epochs = 500

    for model_type in ["LSTM"]:
        print("Model type: ", model_type)
        a_single_sets(model_type)
