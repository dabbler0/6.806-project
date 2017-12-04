import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageEmbedding(nn.Module):
    def __init__(self):
        super(AverageEmbedding, self).__init__()

        # Simple model with no parameters that simply
        # takes average word embedding
        self.linear = nn.Linear(202, 300)

    def forward(self, batch):
        return F.relu(self.linear(batch.mean(1)))

class LSTMAverage(nn.Module):
    def __init__(self):
        super(LSTMAverage, self).__init__()

        self.lstm = nn.LSTM(
            input_size = 202,
            hidden_size = 240,
            num_layers = 1,
            dropout = 0.3,
            #bidirectional = True,
            batch_first = True)

    def forward(self, batch):
        # LSTM
        output, (h, c) = self.lstm(batch)

        # Average LSTM representation
        return output.mean(1)

class LSTMLast(nn.Module):
    def __init__(self):
        super(LSTMLast, self).__init__()

        self.lstm = nn.LSTM(
            input_size = 202,
            hidden_size = 200,
            num_layers = 1,
            batch_first = True)

    def forward(self, batch):
        # LSTM
        output, (h, c) = self.lstm(batch)

        # Final LSTM representation
        return torch.cat([h.squeeze(), c.squeeze()], dim=1)

