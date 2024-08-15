import json
import os
from sklearn.model_selection import KFold
import torch
import numpy as np
from FOVAL_Trainer import FOVAL_Trainer
from RobustVision_Dataset import RobustVision_Dataset
from SimpleLSTM import SimpleLSTM_V2

# ================ Device options
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Use this to print the name of the first device
device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU

# ================ Save folder options
model_save_dir = "../Model"
os.makedirs(model_save_dir, exist_ok=True)

# ================ Objects options
foval_trainer = FOVAL_Trainer()

# ================ Data split options
fixed_splits = []
n_splits = 25
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


def runModel(model_name):
    model_path = os.path.join(model_save_dir, model_name)
    model, hyperparameters = loadModel(model_path)

    l1_lambda = hyperparameters['l1_lambda']
    batch_size = hyperparameters['batch_size']
    embed_dim = hyperparameters['embed_dim']
    learning_rate = hyperparameters['learning_rate']
    weight_decay = hyperparameters['weight_decay']
    l1_lambda = l1_lambda
    dropout_rate = hyperparameters['dropout_rate']
    fc1_dim = hyperparameters['fc1_dim']
    beta = 0.75

    fold_performance = []
    fold_count = 1

    for fixed_split in fixed_splits:
        train_index, val_index, test_index = fixed_split

        if val_index is None:
            username = f"results/{test_index[0]}"
        else:
            username = f"results/{val_index[0]}"

        isExist = os.path.exists(username)
        if not isExist:
            os.makedirs(username)

        foval_trainer.userFolder = username
        print("User folder is set to: ", foval_trainer.userFolder)

        success, goToNextOptimStep, best_val_smae_fold = foval_trainer.runFold(batch_size=batch_size,
                                                                               embed_dim=embed_dim,
                                                                               learning_rate=learning_rate,
                                                                               weight_decay=weight_decay,
                                                                               l1_lambda=l1_lambda,
                                                                               dropoutRate=dropout_rate,
                                                                               fc1_dim=fc1_dim,
                                                                               fold_count=fold_count, n_splits=n_splits,
                                                                               train_index=train_index,
                                                                               val_index=val_index,
                                                                               test_index=test_index,
                                                                               fold_performance=fold_performance,
                                                                               model=model,
                                                                               beta=beta)

        if not success:
            print("Failed")
        else:
            fold_count += 1

    # Calculate overall performance on whole dataset
    best_fold = min(fold_performance, key=lambda x: x['best_val_mae'])
    print(f"Best Fold: {best_fold['fold']} with MAE: {best_fold['best_val_mae']}")
    average_fold_val_mae = np.mean([f['best_val_mae'] for f in fold_performance])
    print(f"Average Validation MAE across folds: {average_fold_val_mae}")


def loadModel(path):
    jsonFile = path + '.json'

    with open(jsonFile, 'r') as f:
        hyper_parameters = json.load(f)

    model = SimpleLSTM_V2(input_size=38, embed_dim=hyper_parameters['embed_dim'],
                          fc1_dim=hyper_parameters['fc1_dim'],
                          dropout_rate=hyper_parameters['dropout_rate']).to(device)

    # BE AWARE IF saved model has already seen some data in training, this is data leakage!
    # Be sure to test only on a separate test set the model hasn't seen before
    # model.load_state_dict(torch.load(modelFile))

    return model, hyper_parameters


if __name__ == '__main__':

    """
        Setup parameters
    """
    sequence_length = 10  # Define your sequence length
    featureCount = 32
    foval_trainer.sequence_length = sequence_length

    # Initialize the data processor
    rv_dataset = RobustVision_Dataset(sequence_length=sequence_length)

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 1. Read in Data
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    aggregated_data = rv_dataset.read_and_aggregate_data()
    foval_trainer.dataset = aggregated_data
    foval_trainer.dataset_obj = rv_dataset

    subject_list = foval_trainer.dataset[
        'SubjectID'].unique()  # Assuming 'subject_list' contains the list of unique subjects

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 2. Define splits we want to use in Cross validation
    #    here we use LOOCV
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    test_subjects_list = subject_list
    for train_subjects, val_subjects in kf.split(test_subjects_list):
        fixed_splits.append((subject_list[train_subjects], subject_list[val_subjects], None))
        # fixed_splits.append((subject_list[train_subjects], None, subject_list[test_subject])) # test

    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    # 3. Define checkpoint and hyperparameters file name from saved model
    #    load model from disc
    # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
    modelName = "foval_chkpt"
    runModel(model_name=modelName)
