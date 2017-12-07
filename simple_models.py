import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from data import word_embedding_size

class AverageEmbedding(nn.Module):
    def __init__(self):
        super(AverageEmbedding, self).__init__()

        # Simple model with no parameters that simply
        # takes average word embedding
        self.linear = nn.Linear(word_embedding_size, 300)

    def forward(self, batch):
        return F.relu(self.linear(batch.mean(1)))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.tan = nn.Tanh()
        self.conv = nn.Conv1d(202, 667, kernel_size=3, padding=1)
        # self.pooling = nn.AvgPool1d(667)


    def forward(self, x):
        x = self.conv(x)
        x = self.tan(x)
        return x

class LSTMAverage(nn.Module):
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def __init__(self,
                dropout = 0.3,
                input_size = word_embedding_size,
                hidden_size = 240,
                bidirectional = True):
        super(LSTMAverage, self).__init__()

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True)

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def signature(self):
        return {
            'type': 'LSTMAverage',
            'dropout': self.dropout,
            'hidden_size': self.hidden_size,
            'bidirectional': self.bidirectional
        }

    def forward(self, batch):
        padding_mask = batch[:, :, 200]

        lengths = padding_mask.sum(1).long()

        lengths, indices = lengths.sort(descending = True)

        # batch x seq_length x word_embedding
        batch = batch[indices.data, :, :]

        # Pack sequence
        packed_sequence = rnn.pack_padded_sequence(
            batch,
            lengths.data.cpu().numpy().tolist(),
            batch_first = True
        )

        # LSTM
        packed_output, (h, c) = self.lstm(packed_sequence)

        # Unpack
        raw_output, _lengths = rnn.pad_packed_sequence(packed_output, batch_first = True)

        # batch is of size (batch) x (seq_length) x (word_embedding)
        # word_embedding[-2] (that's index 200) is going to be equal to 0 when
        # it is a padding sequence.
        # (padding_mask is now of size (batch) x (seq_length))

        # Get mean ignoring things past the end of the
        # sentence.
        output = raw_output.sum(1)
        output = output / lengths.float().unsqueeze(1).expand_as(output)

        # Re-sort output to match the input order.
        _, inverse_indices = indices.sort()
        output = output[inverse_indices.data, :]

        return output

class GRUAverage(nn.Module):
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def __init__(self,
                dropout = 0.3,
                input_size = word_embedding_size,
                hidden_size = 180,
                bidirectional = True):
        super(GRUAverage, self).__init__()

        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True)

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def signature(self):
        return {
            'type': 'GRUAverage',
            'dropout': self.dropout,
            'hidden_size': self.hidden_size,
            'bidirectional': self.bidirectional
        }

    def forward(self, batch):
        padding_mask = batch[:, :, 200]

        lengths = padding_mask.sum(1).long()

        lengths, indices = lengths.sort(descending = True)

        # batch x seq_length x word_embedding
        batch = batch[indices.data, :, :]

        # Pack sequence
        packed_sequence = rnn.pack_padded_sequence(
            batch,
            lengths.data.cpu().numpy().tolist(),
            batch_first = True
        )

        # LSTM
        packed_output, h = self.gru(packed_sequence)

        # Unpack
        raw_output, _lengths = rnn.pad_packed_sequence(packed_output, batch_first = True)

        # batch is of size (batch) x (seq_length) x (word_embedding)
        # word_embedding[-2] (that's index 200) is going to be equal to 0 when
        # it is a padding sequence.
        # (padding_mask is now of size (batch) x (seq_length))

        # Get mean ignoring things past the end of the
        # sentence.
        output = raw_output.sum(1)
        output = output / lengths.float().unsqueeze(1).expand_as(output)

        # Re-sort output to match the input order.
        _, inverse_indices = indices.sort()
        output = output[inverse_indices.data, :]

        return output
