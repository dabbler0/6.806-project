import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

import math

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

def AttentionIsAllYouNeed(nn.Module):
    def __init__(self, heads, output_size = 200, key_size = 64, value_size = 64):
        self.heads = heads
        self.output_size = output_size
        self.key_size = key_size
        self.value_size = value_size

        # The initial transform, which generates
        # all the queries, keys, and values.
        self.initial_transform = nn.Linear(
            202,
            self.heads * (2 * key_size + value_size)
        )

        # The merge transform, which takes the multiple
        # attention heads and produces the output encoding
        self.merge_transform = nn.Linear(
            self.heads * value_size,
            self.output_size
        )

        self.frequency_matrix = torch.pow(
            1e-4, torch.arange(1, 11) / 10).view(1, -1)

    def output_size(self):
        return self.output_size

    def signature(self):
        return {
            'type': 'AttenionIsAllYouNeed',
            'output_size': self.output_size,
            'heads': self.heads,
            'positional_encodings': 10,
            'key_size': self.key_size,
            'value_size': self.value_size
        }

    def forward(self, batch):
        # batch will be of size (batch_size) x (seq_length) x (embedding_size)
        # We will perform a self-attention embedding.
        # We want to compute (head) keys, values and queries for each word.

        padding_mask = batch[:, :, 200]

        key_size, value_size, heads = self.key_size, self.value_size, self.heads

        batch_size, seq_length, embedding_size = batch.size()
        pos_size = self.frequency_matrix.size()[1]

        # Positional encoding
        positions = torch.arange(0, seq_length).view(
            1, seq_length, 1)
        position_ratios = torch.torch.mm(positions, self.frequency_matrix))
        sin_encoding = torch.sin(position_ratios).expand(batch_size, seq_length, pos_size)
        cos_encoding = torch.cos(position_ratios).expand(batch_size, seq_length, pos_size)
        batch = torch.cat([batch, sin_encoding, cos_encoding], dim = 3)

        # Update embedding size to match embedding with positional encoding
        embedding_size += pos_size * 2

        # Get queries, keys, values
        all_info = self.initial_transform(
            batch.view(-1, embedding_size)).view(batch_size, seq_length, -1)

        # Each of these should be (batch_size) x (seq_length) x (heads) x (dim)
        queries = all_info[:, :, :(self.heads * key_size)]
        keys = all_info[:, :, (self.heads * key_size):(self.heads * key_size) * 2]
        values = all_info[:, :, -(self.heads * value_size):]

        queries = queries.view(batch_size, seq_length, heads, key_size)
        keys = queries.view(batch_size, seq_length, heads, key_size)
        values = queries.view(batch_size, seq_length, heads, value_size)

        # Batched matrix multiplication to get query-key similarity.
        # We need to transpose queries and keys to put heads first,
        # so that we can batch them properly.
        queries = queries.transpose(1, 2).copy()
        keys = queries.transpose(1, 2).copy()
        values = queries.transpose(1, 2).copy()

        # Get attentions
        attentions = torch.bmm(
            queries.view(-1, seq_length, key_size),
            keys.view(-1, seq_length, key_size)
        ).view(batch_size, heads, seq_length, seq_length)

        # "Scaled dot-product attention"
        attentions /= math.sqrt(key_size)

        # Softmax over the last dimension
        attentions = F.softmax(attentions, dim = 3)

        # Mask out padding vectors
        attentions *= padding_mask.unsqueeze(1).unsqueeze(1).expand_as(attentions)
        attentions /= attentions.sum(dim = 3).unsqueeze(3).expand_as(attentions)

        # Apply attention to values
        # This should produce a matrix of size (batch_size) x (heads) x (seq_len) x (value_dim)
        aggregations = torch.bmm(
            attentions.view(-1, seq_length, seq_length),
            values.view(-1, seq_length, value_size)
        ).view(batch_size, heads, seq_length, value_size)

        # Re-concatenate all of the aggregations
        aggregations = aggregations.transpose(1, 2).copy().view(batch_size, seq_length, -1)

        # Merge transform, yielding (batch_size) x (seq_len) x (embedding_size)
        embeddings = self.merge_transform(
            aggregations.view(batch_size * seq_length, -1)).view(batch_size, seq_length, -1)

        # Average embedding
        return (
            (embeddings * padding_mask.unsqueeze(2).expand_as(embeddings)).sum(1) /
            padding_mask.sum(1).unsqueeze(1).expand(batch_size, self.output_size)
        )
