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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.tan = nn.Tanh()
        self.conv = nn.Conv1d(202, 667, kernel_size=3, padding=1)
        # self.pooling = nn.AvgPool1d(667)

    def forward(self, x):
        x = x.transpose(1, 2)
        padding_mask = x[:, 200, :]
        x = self.conv(x)
        x = self.tan(x)
        x = (padding_mask.unsqueeze(2)*x.transpose(1, 2)).sum(1)/padding_mask.sum(1).unsqueeze(1)
        return x

class LSTMAverage(nn.Module):
    def __init__(self):
        super(LSTMAverage, self).__init__()

        self.lstm = nn.LSTM(
            input_size = 202,
            hidden_size = 240,
            num_layers = 1,
            dropout = 0.7,
            #bidirectional = True,
            batch_first = True)

    def forward(self, batch):
        # LSTM
        output, (h, c) = self.lstm(batch)

        # batch is of size (batch) x (seq_length) x (word_embedding)
        # word_embedding[-2] (that's index 200) is going to be equal to 0 when
        # it is a padding sequence.
        padding_mask = batch[:, :, 200]
        # (padding_mask is now of size (batch) x (seq_length))

        # Get mean ignoring things past the end of the
        # sentence.
        return (padding_mask.unsqueeze(2) * output).sum(1) / padding_mask.sum(1).unsqueeze(1)

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
