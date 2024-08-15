# FOVAL: Focal Depth Estimation

This project implements a model for focal depth estimation. The model is trained using the Robust Vision dataset and leverages a simple LSTM architecture - called FOVAL.
 The project includes scripts for training the model, evaluating its performance, and saving model activations, which can be utilized for further analysis or in related projects like ACTIF.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Detailed Code Description](#description)
5. [Project Structure](#project-structure)
6. [License](#license)
7. [Contributing](#contributing)
8. [Contact](#contact)

## 1. Introduction

FOVAL is focused on estimating focal depth using a sequence of visual data points. The core of the project revolves around an LSTM-based model that processes temporal sequences of features derived from visual data. The model is trained to predict the depth from the input features, and its performance is evaluated across multiple cross-validation folds. The project also includes functionalities to save intermediate activations of the model, which can be used for feature importance ranking or other analytical purposes.
## 2. Installation

### Prerequisites

- Anaconda/Miniconda
- Python 3.x

### Environment Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/benedikt-hosp/foval.git
   cd FOVAL
   ```
   
2. **Setup the conda environment:**
```
conda env create -f environment.yml
conda activate foval_env
```
## 3. Usage
### Running the Model
#### 1. Train and evaluate the model:

To train and evaluate the LSTM model on the focal depth estimation task, run the following command:
```
python src/main.py
```
This script will:

- Load and preprocess the dataset.
- Define the LSTM model architecture.
- Split the data into training, validation, and test sets using cross-validation.
- Train the model across multiple epochs.
- Evaluate the model's performance on the validation and test sets.
- Save the best model's state and activations for further analysis.

#### 2. Monitor the training process:

Training progress, including loss and accuracy metrics, will be logged and can be monitored. After training, results, including model weights and performance metrics, are saved in the results/ directory.

### Saving and Using Model Activations
The model is configured to save intermediate activations during the validation phase. These activations can be used for tasks such as feature importance analysis or visualizing how the model processes the input data.

### Project Structure
data/: Contains data files or datasets.

notebooks/: Jupyter notebooks for exploratory data analysis and experiments.

src/: Source code, including feature extraction, model training, and evaluation scripts.

	- FOVAL_Trainer.py: Handles the training process, including data loading, model training, validation, and saving results.

	- RobustVision_Dataset.py: Contains functions for data preprocessing, feature extraction, and sequence generation.

	- SimpleLSTM.py: Defines the LSTM-based model used for focal depth estimation.

	- Utilities.py: Includes utility functions for data handling, model definition, and results analysis.

	- main.py: The main script that integrates all components to run the training and evaluation pipeline.
- models/: Directory to store pre-trained or saved models.
- results/: Directory where output results, logs, and saved activations are stored.
- environment.yml: Conda environment setup file.

## 4. Detailed Code Description
#### FOVAL_Trainer.py
This script manages the entire training process:

- **Data Loading and Preprocessing:** Loads the dataset, applies transformations, and creates sequences of data.
- **Model Training:** Trains the LSTM model over multiple epochs and evaluates performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and other metrics.
- **Cross-Validation:** Implements K-Fold cross-validation to ensure robust performance evaluation.
- **Saving Results:** Saves the best model's weights and intermediate activations for further analysis.
#### RobustVision_Dataset.py
This module handles all aspects of data management:

- **Data Reading and Aggregation:** Reads and aggregates data from multiple subjects in the dataset.

- **Feature Creation:** Extracts relevant features from raw data.

- **Data Transformation:** Applies various transformations to normalize and scale the data.

- **Sequence Generation:** Generates sequences of data points for use in the LSTM model.

#### SimpleLSTM.py
Defines the architecture of the LSTM model:

- **LSTM Layer:** Processes sequences of input features to capture temporal dependencies.
- **Fully Connected Layers:** Maps the LSTM outputs to the final depth predictions.
- **Dropout and Normalization:** Implements regularization techniques to improve generalization.

#### Utilities.py
Includes various utility functions:

- **Data Handling:** Functions for creating data loaders and tensors.
- **Model Definition:** Functions for setting up the LSTM model and its optimizer.
- **Results Analysis:** Functions to analyze and visualize model predictions and errors

## 5. License
This project is licensed under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license. You are free to share and adapt the material as long as appropriate credit is given.

## 6. Contributing
Contributions are welcome! Please open an issue or submit a pull request for any bug fixes, features, or improvements.




