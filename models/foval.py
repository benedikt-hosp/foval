import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn


class Foval(nn.Module):
    def __init__(self, device, feature_count, model_type="LSTM"):
        super(Foval, self).__init__()

        self.pos_encoding = None
        self.project_to_embed = None
        self.linear_projection = None
        self.projection = None
        self.device = device
        self.model_type = model_type  # Choose between 'LSTM', 'GRU', 'CNN', 'Attention', 'TCN'
        self.feature_count = feature_count
        self.input_size = 38
        self.hidden_layer_size = None
        self.fc1_dim = None
        self.dropout_rate = None
        self.outputsize = 1
        self.embed_dim = None
        self.seq_len = 10
        self.num_heads = 5

        # Layers
        self.input_linear = None
        self.model = None
        self.fc1 = None
        self.fc5 = None
        self.dropout = None
        self.activation = None
        self.batchnorm = None
        self.layernorm = None

    def initialize(self, input_size, hidden_layer_size, fc1_dim, dropout_rate):
        input_size = 34

        self.input_size = input_size
        self.embed_dim = hidden_layer_size
        self.fc1_dim = fc1_dim
        self.dropout_rate = dropout_rate
        # Shared input projection
        self.input_linear = nn.Linear(self.input_size, self.input_size)

        # Optional projection for attention
        if self.model_type == "Attention":
            self.project_to_embed = nn.Linear(self.input_size, self.embed_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim))
            self.model = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
            self.layernorm = nn.LayerNorm(self.embed_dim)
        elif self.model_type == "LSTM":
            self.model = nn.LSTM(input_size=self.input_size, hidden_size=self.embed_dim, batch_first=True)
            self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        elif self.model_type == "GRU":
            self.model = nn.GRU(input_size=self.input_size, hidden_size=self.embed_dim, batch_first=True)
            self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        elif self.model_type == "CNN":
            self.model = nn.Conv1d(in_channels=self.input_size, out_channels=self.embed_dim, kernel_size=3, padding=1)
            self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        elif self.model_type == "TCN":
            self.model = nn.Sequential(
                nn.Conv1d(in_channels=self.input_size, out_channels=self.embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=3, padding=1)
            )
            self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Fully connected head
        self.fc1 = nn.Linear(self.embed_dim, self.fc1_dim // 4)
        self.fc5 = nn.Linear(self.fc1_dim // 4, self.outputsize)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.to(self.device)

    def forward(self, input_seq, return_intermediates=True):
        input_activations = self.input_linear(input_seq)

        if self.model_type == "Attention":
            projected = self.project_to_embed(input_seq)  # shape: (B, T, D)
            embedded = projected + self.pos_encoding[:, :self.seq_len, :]  # Add position info

            # MultiheadAttention
            output, _ = self.model(embedded, embedded, embedded)  # (B, T, D)
            output = self.layernorm(output)
        elif self.model_type in ["LSTM", "GRU"]:
            output, _ = self.model(input_seq)  # (B, T, D)
            output = output.permute(0, 2, 1)  # (B, D, T)
            output = self.batchnorm(output)
            output = output.permute(0, 2, 1)  # (B, T, D)
        elif self.model_type in ["CNN", "TCN"]:
            output = self.model(input_seq.permute(0, 2, 1))  # (B, D, T)
            output = self.batchnorm(output)
            output = output.permute(0, 2, 1)  # (B, T, D)

        # Global temporal pooling
        output, _ = output.max(dim=1)  # (B, D)

        # Fully connected layers
        output = self.dropout(output)
        output = self.activation(self.fc1(output))
        predictions = self.fc5(output)

        predictions = torch.nan_to_num(predictions, nan=0.0, posinf=9.0, neginf=0.0)

        if return_intermediates:
            return predictions, {
                'Input_activations': input_activations,
                'Model_output': output,
                'Predictions': predictions
            }
        else:
            return predictions


# # Original
# class Foval(nn.Module):
#     def __init__(self, device, feature_count):
#         super(Foval, self).__init__()
#
#         self.device = device
#
#         # Hyperparameteres
#         self.hidden_layer_size = None
#         self.feature_count = feature_count
#         self.input_size = 34
#         self.embed_dim = None
#         self.fc1_dim = None
#         self.fc5_dim = None
#         self.outputsize = 1
#         self.dropout_rate = None
#
#         # Layers
#         self.input_linear = None
#         self.lstm = None
#         self.layernorm = None
#         self.batchnorm = None
#         self.fc1 = None
#         self.fc5 = None
#         self.activation = None
#         self.dropout = None
#
#         # Load Hyperparameteres from file
#
#     def initialize(self, input_size, hidden_layer_size, fc1_dim, dropout_rate):
#
#         input_size = 34
#         # print("HP: ", input_size, hidden_layer_size, fc1_dim, dropout_rate)
#         # Linear layer to transform input features if needed
#         self.input_linear = nn.Linear(in_features=input_size, out_features=input_size)
#
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size=input_size, num_layers=1, batch_first=True, hidden_size=hidden_layer_size)
#         self.layernorm = nn.LayerNorm(hidden_layer_size)
#         self.batchnorm = nn.BatchNorm1d(hidden_layer_size)
#
#         # Additional fully connected layers
#         self.fc1 = nn.Linear(hidden_layer_size, fc1_dim // 4)  # Use integer division
#         self.fc5 = nn.Linear(fc1_dim // 4, self.outputsize)  # Final FC layer for output
#         self.activation = nn.ELU()
#
#         # Dropout layer
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.to(self.device)
#
#     def forward(self, input_seq, return_intermediates=True):
#
#         # Capture the input activations after applying the linear transformation
#         input_activations = self.input_linear(input_seq)
#
#         # Pass the activations through the LSTM layer
#         lstm_out, _ = self.lstm(input_seq)
#
#         # Permute and apply batch normalization
#         lstm_out_1 = lstm_out.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_length)
#         lstm_out_2 = self.batchnorm(lstm_out_1)
#         lstm_out_3 = lstm_out_2.permute(0, 2, 1)  # Change back to (batch_size, seq_length, num_features)
#
#         # Max pooling over the time dimension
#         lstm_out_max_timestep, _ = lstm_out_3.max(dim=1)  # Max-pooling over time
#         lstm_dropout = self.dropout(lstm_out_max_timestep)
#         fc1_out = self.fc1(lstm_dropout)
#         fc1_elu_out = self.activation(fc1_out)
#         predictions = self.fc5(fc1_elu_out)
#
#         if return_intermediates:
#             intermediates = {'input_seq': input_seq, 'Input_activations': input_activations,
#                              'Input_Weights': self.input_linear.weight.data.cpu().numpy(), 'LSTM_Out': lstm_out,
#                              'LSTM_Weights_IH': self.lstm.weight_ih_l0.data.cpu().numpy(),
#                              'LSTM_Weights_HH': self.lstm.weight_hh_l0.data.cpu().numpy(),
#                              'Max_Timestep': lstm_out_max_timestep, 'FC1_Out': fc1_out,
#                              'FC1_Weights': self.fc1.weight.data.cpu().numpy(), 'FC1_ELU_Out': fc1_elu_out,
#                              'Output': predictions, 'FC5_Weights': self.fc5.weight.data.cpu().numpy()}
#
#             # Save the weight matrices for the LSTM layer
#
#             # Save the weight matrix for the first fully connected layer
#
#             # Save the weight matrix for the final fully connected layer
#             # Save the weight matrix of the first linear layer
#
#             return predictions, intermediates
#         else:
#             return predictions
