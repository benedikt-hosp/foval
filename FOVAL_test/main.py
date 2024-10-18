import random
import torch
from sklearn.cluster import KMeans

from data.TuftsDataset import TuftsDataset
from data.foval_preprocessor import input_features
from data.giw_dataset import GIWDataset
from data.robustVision_dataset import RobustVisionDataset

from training.foval_trainer import FOVALTrainer

import scipy.io
import scipy.io
import scipy.io
import numpy as np
import pandas as pd
import scipy.io
import os
import shutil

# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))  # Use this to print the name of the first device
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU


def setup_device():
    """Sets up the GPU device."""
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda:0")


def seed_everything(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_model_directory(model_save_dir="models"):
    """Creates a directory for saving models if it doesn't exist."""
    os.makedirs(model_save_dir, exist_ok=True)
    return model_save_dir


def estimate_dynamic_gaze_origin(pog, dir_L, dir_R):
    # Ensure that POG, dir_L, and dir_R are numpy arrays with 3 elements each
    if isinstance(pog, np.float64) or isinstance(dir_L, np.float64) or isinstance(dir_R, np.float64):
        raise ValueError("POG, dir_L, or dir_R is not a vector. Expected 3D vectors.")

    if pog.shape != (3,) or dir_L.shape != (3,) or dir_R.shape != (3,):
        raise ValueError(
            f"POG or direction vectors are not 3D: POG: {pog.shape}, dir_L: {dir_L.shape}, dir_R: {dir_R.shape}")

    # Create the system of linear equations to solve for the origins
    A = np.vstack([dir_L, -dir_R]).T
    b = pog
    dist_vector, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Compute the origins of the left and right gaze
    origin_L = -dist_vector[0] * dir_L
    origin_R = dist_vector[1] * dir_R

    return origin_L, origin_R



def process_mat_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all .mat files in the input folder
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

    for mat_file in mat_files:
        print(f"Processing {mat_file}:")

        # Load the .mat file
        mat_data = scipy.io.loadmat(os.path.join(input_folder, mat_file))

        # Extract ProcessData from the .mat file
        ProcessData = mat_data.get('ProcessData', {})

        if ProcessData is None or not ProcessData:
            print(f"Skipping {mat_file} due to missing 'ProcessData'.")
            continue

        try:
            # Extract the time data
            T = ProcessData['T'].flatten()  # Time
            T_list = T[0].flatten().tolist()

            # Extract the scene depth data
            SceneDepth = ProcessData['SceneDepth'].flatten()  # Ground truth depth
            Sc_list = (SceneDepth[0].flatten() / 1000).tolist()  # Convert mm to cm by dividing by 10

            # Gaze vectors for left and right eyes
            Gaze_Vector_L = ProcessData['ETG'][0][0]['EIHvector_0'][0][0]  # Left eye gaze direction
            Gaze_Vector_R = ProcessData['ETG'][0][0]['EIHvector_1'][0][0]  # Right eye gaze direction

            # Extract Vergence Angle
            Vergence_Angle = ProcessData['ETG'][0][0]['Vergence'][0][0].flatten()
            # print(Vergence_Angle)
            Vergence_Angle_list = pd.to_numeric(Vergence_Angle, errors='coerce').tolist()  # Convert to numeric

            # Initialize lists for columns to store valid data or NaN
            valid_times = []
            valid_scene_depths = []
            valid_gaze_direction_L_x = []
            valid_gaze_direction_L_y = []
            valid_gaze_direction_L_z = []
            valid_gaze_direction_R_x = []
            valid_gaze_direction_R_y = []
            valid_gaze_direction_R_z = []
            valid_vergence_angles = []

            # Iterate over each sample and check if the data is valid or NaN
            for i in range(len(T_list)):
                try:
                    # Append time value or NaN
                    valid_times.append(T_list[i] if i < len(T_list) else np.nan)

                    # Append scene depth or NaN
                    valid_scene_depths.append(Sc_list[i] if i < len(Sc_list) else np.nan)

                    # Append gaze direction vectors for both eyes or NaN for missing values
                    valid_gaze_direction_L_x.append(Gaze_Vector_L[i, 0] if i < len(Gaze_Vector_L) else np.nan)
                    valid_gaze_direction_L_y.append(Gaze_Vector_L[i, 1] if i < len(Gaze_Vector_L) else np.nan)
                    valid_gaze_direction_L_z.append(Gaze_Vector_L[i, 2] if i < len(Gaze_Vector_L) else np.nan)

                    valid_gaze_direction_R_x.append(Gaze_Vector_R[i, 0] if i < len(Gaze_Vector_R) else np.nan)
                    valid_gaze_direction_R_y.append(Gaze_Vector_R[i, 1] if i < len(Gaze_Vector_R) else np.nan)
                    valid_gaze_direction_R_z.append(Gaze_Vector_R[i, 2] if i < len(Gaze_Vector_R) else np.nan)

                    # Append vergence angle or NaN
                    valid_vergence_angles.append(Vergence_Angle_list[i] if i < len(Vergence_Angle_list) else np.nan)

                except IndexError as e:
                    print(f"Index error at sample {i} in {mat_file}: {e}")
                    valid_times.append(np.nan)
                    valid_scene_depths.append(np.nan)
                    valid_gaze_direction_L_x.append(np.nan)
                    valid_gaze_direction_L_y.append(np.nan)
                    valid_gaze_direction_L_z.append(np.nan)
                    valid_gaze_direction_R_x.append(np.nan)
                    valid_gaze_direction_R_y.append(np.nan)
                    valid_gaze_direction_R_z.append(np.nan)
                    valid_vergence_angles.append(np.nan)

            data_dict = {
                'Time': valid_times,
                'Gt_Depth': valid_scene_depths,  # Updated to 'Gt_Depth'
                'World_Gaze_Direction_L_X': valid_gaze_direction_L_x,  # Updated name
                'World_Gaze_Direction_L_Y': valid_gaze_direction_L_y,  # Updated name
                'World_Gaze_Direction_L_Z': valid_gaze_direction_L_z,  # Updated name
                'World_Gaze_Direction_R_X': valid_gaze_direction_R_x,  # Updated name
                'World_Gaze_Direction_R_Y': valid_gaze_direction_R_y,  # Updated name
                'World_Gaze_Direction_R_Z': valid_gaze_direction_R_z,  # Updated name
                'Vergence_Angle': valid_vergence_angles  # Added vergence angle
            }

            # Create a DataFrame from the collected data
            df = pd.DataFrame(data_dict)

            # Save DataFrame to a CSV file in the output folder
            output_csv_file = os.path.join(output_folder, f"{mat_file.replace('.mat', '.csv')}")
            df.to_csv(output_csv_file, index=False, sep='\t')  # Using tab as separator
            print(f"Processed {mat_file} and saved to {output_csv_file}")

        except KeyError as e:
            print(f"Key error when processing {mat_file}: {e}")
            continue
        except IndexError as e:
            print(f"Index error when accessing {mat_file}: {e}")
            continue



def delete_folder_contents(folder_path):
    # Ensure the folder exists
    if os.path.exists(folder_path):
        # Iterate over all contents in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file or a folder
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the folder and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def process_gaze360_mat_data_fixed(mat_data, output_folder):
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_data)

    # Extract relevant data fields
    valid_times = mat_data['ts'].flatten()

    # Extract the 3D positions for target and eyes
    target_pos3d_x = mat_data['target_pos3d'][:, 0]
    target_pos3d_y = mat_data['target_pos3d'][:, 1]
    target_pos3d_z = mat_data['target_pos3d'][:, 2]

    person_eyes3d_x = mat_data['person_eyes3d'][:, 0]
    person_eyes3d_y = mat_data['person_eyes3d'][:, 1]
    person_eyes3d_z = mat_data['person_eyes3d'][:, 2]

    # Calculate Euclidean distance (depth) between the eye center and the target
    valid_scene_depths = np.sqrt(
        (target_pos3d_x - person_eyes3d_x) ** 2 +
        (target_pos3d_y - person_eyes3d_y) ** 2 +
        (target_pos3d_z - person_eyes3d_z) ** 2
    )

    # Gaze directions (these could be from left and right eyes, update if necessary)
    valid_gaze_direction_L_x = mat_data['gaze_dir'][:, 0].flatten()
    valid_gaze_direction_L_y = mat_data['gaze_dir'][:, 1].flatten()
    valid_gaze_direction_L_z = mat_data['gaze_dir'][:, 2].flatten()

    valid_gaze_direction_R_x = mat_data['gaze_dir'][:, 0].flatten()
    valid_gaze_direction_R_y = mat_data['gaze_dir'][:, 1].flatten()
    valid_gaze_direction_R_z = mat_data['gaze_dir'][:, 2].flatten()

    # Extract World Gaze Origins (Eye origins for left and right eyes)
    world_gaze_origin_L_x = mat_data['person_eyes3d'][:, 0]
    world_gaze_origin_L_y = mat_data['person_eyes3d'][:, 1]
    world_gaze_origin_L_z = mat_data['person_eyes3d'][:, 2]

    world_gaze_origin_R_x = mat_data['person_eyes3d'][:, 0]
    world_gaze_origin_R_y = mat_data['person_eyes3d'][:, 1]
    world_gaze_origin_R_z = mat_data['person_eyes3d'][:, 2]

    # Create the final dictionary including gaze origins
    data_dict = {
        'Time': valid_times,
        'Gt_Depth': valid_scene_depths,
        'World_Gaze_Direction_L_X': valid_gaze_direction_L_x,
        'World_Gaze_Direction_L_Y': valid_gaze_direction_L_y,
        'World_Gaze_Direction_L_Z': valid_gaze_direction_L_z,
        'World_Gaze_Direction_R_X': valid_gaze_direction_R_x,
        'World_Gaze_Direction_R_Y': valid_gaze_direction_R_y,
        'World_Gaze_Direction_R_Z': valid_gaze_direction_R_z,
        'World_Gaze_Origin_R_X': world_gaze_origin_R_x,
        'World_Gaze_Origin_R_Y': world_gaze_origin_R_y,
        'World_Gaze_Origin_R_Z': world_gaze_origin_R_z,
        'World_Gaze_Origin_L_X': world_gaze_origin_L_x,
        'World_Gaze_Origin_L_Y': world_gaze_origin_L_y,
        'World_Gaze_Origin_L_Z': world_gaze_origin_L_z,
    }

    # Convert to DataFrame
    df = pd.DataFrame(data_dict)

    # Extract the person_identity field to determine participants
    person_identities = mat_data['person_identity'].flatten()

    # Combine the previously extracted data with person identities into the DataFrame
    df['person_identity'] = person_identities

    # Use Gaze Origins for clustering
    gaze_origins = df[['World_Gaze_Origin_L_X', 'World_Gaze_Origin_L_Z', 'World_Gaze_Origin_R_X', 'World_Gaze_Origin_R_Z']].values

    # Step 1: Apply KMeans clustering to merge participants into 238 groups
    kmeans = KMeans(n_clusters=238, random_state=42)
    df['clustered_participant'] = kmeans.fit_predict(gaze_origins)

    # Get the unique clustered participants
    unique_clustered_participants = np.unique(df['clustered_participant'])

    # Clear the output folder before saving new files
    clear_output_folder(output_folder)

    # Step 2: Create a separate file for each unique clustered participant
    for clustered_participant in unique_clustered_participants:
        # Filter the data for the current clustered participant
        participant_data = df[df['clustered_participant'] == clustered_participant]

        # Check if there are enough samples with Gt_Depth < 300 (3 meters)
        num_samples_under_3m = participant_data[participant_data['Gt_Depth'] < 3].shape[0]

        if num_samples_under_3m > 0:  # If there are any samples with Gt_Depth < 3 meters
            # Define the output file path
            output_file = f"{output_folder}participant_{int(clustered_participant)}.csv"

            # Save the filtered DataFrame to a CSV file using tab separator
            participant_data.to_csv(output_file, index=False, sep="\t")

            print(f"Saved data for clustered participant {int(clustered_participant)} to {output_file}")
        else:
            print(f"Skipped participant_{int(clustered_participant)}: No depths under 3 meters.")


def parseGIWRawData(folder):
    trials = {
        "Trial 1": './data/input/gaze_in_wild/T1_indoor_walk/',
        "Trial 2": './data/input/gaze_in_wild/T2_ball_catch/',
        "Trial 3": './data/input/gaze_in_wild/T3_visual_search/',
        "Trial 4": './data/input/gaze_in_wild/T4_tea_making/'
    }

    for trial_name, trial_path in trials.items():
        print(f"\n\nWorking on {trial_name}")

        # Define input and output folders
        input_folder = trial_path + folder
        output_folder = trial_path + 'parsed_to_csv'

        # First, delete old files from the output folder
        delete_folder_contents(output_folder)

        # Process the .mat files and export CSVs
        process_mat_files(input_folder, output_folder)


def clear_output_folder(output_folder):
    # Check if the output folder exists
    if os.path.exists(output_folder):
        # Remove all contents of the output folder
        shutil.rmtree(output_folder)

    # Recreate the empty folder
    os.makedirs(output_folder, exist_ok=True)


def process_tufts_data(input_file, output_folder):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Map the current columns to the desired format
    df_transformed = pd.DataFrame({
        'Gt_Depth': df['distance'],  # Ground truth depth
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

    # Number of subjects (splits)
    num_subjects = 8

    # 20 split
    # Fold 20 MAE: 21.386478424072266
    # Average Validation MAE across folds: 26.373850297927856
    # Best Fold with MAE: 9.68142032623291
    # Average Cross-Validation MSE: 26.373850297927856
    # Model Average Mean Absolute Error: 26.373850297927856

    # Split the data into 10 approximately equal parts without shuffling
    split_data = np.array_split(df_transformed, num_subjects)

    # Output folder for the files
    os.makedirs(output_folder, exist_ok=True)

    delete_folder_contents(output_folder)

    # Save each part to a separate file as a different subject
    for i, subject_data in enumerate(split_data):
        output_file = os.path.join(output_folder, f'subject_{i + 1}.csv')
        subject_data.to_csv(output_file, index=False, sep="\t")
        print(f"Saved data for Subject {i + 1} to {output_file}")

def main():
    seed_everything(seed=42)
    device = setup_device()
    n_epochs = 300

    # Only needed once

    '''
    Prepare GIW dataset
    '''
    # folder = 'raw'
    # folder = 'raw_processed'
    # parseGIWRawData(folder)

    '''
    Prepare Gaze360 dataset
    As there are over 1000 person identifications but reportedly only 238 participants took part,
    we need to merge the data somehow. We use DBSCAN on the eye origin data to cluster samples into 238 subjects.
    While this is not ideal, with no information at all it is better to cluster data than to assume 1000 subjects.
    As soon as we get information about how person identification is related to the 238 subjects, we can optimize it.
    For now , we need to assume this.
    '''
    # Process and save the data for each participant
    # mat_file = 'data/input/gaze360/raw/metadata.mat'
    # output_folder = 'data/input/gaze360/parsed_to_csv/'
    # process_gaze360_mat_data_fixed(mat_file, output_folder)

    '''
     Prepare Tufts Dataset
    
     '''
    # Process and save the data for each participant
    # input_file = 'data/input/tufts/raw/input_raw.csv'
    # output_folder = 'data/input/tufts/parsed_to_csv/'
    # process_tufts_data(input_file, output_folder)

    # Load and prepare dataset
    # dataset = RobustVisionDataset(data_dir="data/input/robustvision/")                    # 9.1 avg
    dataset = GIWDataset(data_dir="data/input/gaze_in_wild/", trial_name="T4_tea_making")
    # dataset = Gaze360Dataset(data_dir="data/input/gaze360/", test_split_size=10)
    # dataset = TuftsDataset(data_dir="data/input/tufts/", test_split_size=10)

    dataset.load_data()

    # Initialize the FOVAL trainer
    foval_trainer = FOVALTrainer(config_path="models/config/foval.json", dataset=dataset, device=device,
                                 feature_names=input_features, save_intermediates_every_epoch=False)
    foval_trainer.setup()

    mean_mae = foval_trainer.cross_validate(num_epochs=n_epochs)
    print(f"Model Average Mean Absolute Error: {mean_mae}")


if __name__ == "__main__":
    main()
