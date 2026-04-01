import torch
import torch.nn as nn


class Foval(nn.Module):
    def __init__(self, device, feature_count, model_type="LSTM"):
        super(Foval, self).__init__()

        self.device = device
        self.model_type = model_type
        self.feature_count = feature_count
        self.input_size = 38
        self.hidden_layer_size = None
        self.fc1_dim = None
        self.dropout_rate = None
        self.outputsize = 1
        self.embed_dim = None
        self.seq_len = 10
        self.num_heads = 5

        # Layers (initialized in initialize())
        self.pos_encoding = None
        self.project_to_embed = None
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

        self.input_linear = nn.Linear(self.input_size, self.input_size)

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

        self.fc1 = nn.Linear(self.embed_dim, self.fc1_dim // 4)
        self.fc5 = nn.Linear(self.fc1_dim // 4, self.outputsize)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.to(self.device)

    def forward(self, input_seq, return_intermediates=True):
        input_activations = self.input_linear(input_seq)

        if self.model_type == "Attention":
            projected = self.project_to_embed(input_seq)
            embedded = projected + self.pos_encoding[:, :self.seq_len, :]
            output, _ = self.model(embedded, embedded, embedded)
            output = self.layernorm(output)
        elif self.model_type in ["LSTM", "GRU"]:
            output, _ = self.model(input_seq)
            output = output.permute(0, 2, 1)
            output = self.batchnorm(output)
            output = output.permute(0, 2, 1)
        elif self.model_type in ["CNN", "TCN"]:
            output = self.model(input_seq.permute(0, 2, 1))
            output = self.batchnorm(output)
            output = output.permute(0, 2, 1)

        # Global temporal pooling
        output, _ = output.max(dim=1)

        # Fully connected head
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
        return predictions
