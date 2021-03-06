import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

import math

from data import word_embedding_size

class CNN(nn.Module):
    '''
    Convolutional neural net, filter size 3.
    '''
    def __init__(self, hidden_size = 667, dropout = 0.3, input_size = 202):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pad_index = input_size - 2
        self.conv = nn.Conv1d(input_size, self.hidden_size, kernel_size=3, padding=1)

    def forward(self, x, padding_mask):
        print type(padding_mask)
        padding_mask = padding_mask.float()
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        print type(padding_mask), type(x)
        x = (padding_mask.unsqueeze(2)*x.transpose(1, 2)).sum(1)/padding_mask.sum(1).unsqueeze(1)
        return x

    def output_size(self):
        return self.hidden_size

    def signature(self):
        return {
            'type': 'CNN',
            'hidden_size': self.hidden_size
        }

class GRUDecoder(nn.Module):
    '''
    Decoder for Body-to-Title summarization network,
    which generates the title.
    '''
    def __init__(self,
                    embedding_layer,
                    dropout = 0.3,
                    input_size = 302,
                    hidden_size = 100,
                    output_size = 10000):

        super(GRUDecoder, self).__init__()
        self.gru = nn.GRU(
            dropout = dropout,
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 1,
            batch_first = True,
        )
        self.embedding_layer = embedding_layer
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.initial_dropout = nn.Dropout(dropout)

        self.output = nn.Linear(hidden_size, output_size)
        self.vocab_size = output_size

    def forward(self, encodings, targets):
        # Pack the targets
        mask = targets.gt(0)
        lengths = mask.long().sum(1)

        lengths, indices = torch.sort(lengths, descending = True)
        targets = targets[indices]

        # Targets here should be a list of indices.
        # Right-shift the targets by one
        shifted_targets = torch.cat(
            [
                Variable(torch.zeros((targets.size()[0], 1)).long().cuda()),
                targets
            ],
            dim = 1
        )[:, :-1]

        embedded = self.embedding_layer(shifted_targets)

        encoding_size = encodings.size()[1]

        # The intermediate vector representation will be fed
        # again to the network at every timestep
        # (this was done to encourage gradients to flow through the
        # intermediate representation when the ordinary encoder-decoder
        # model didn't work. This didn't work either though).
        packed_sequence = rnn.pack_padded_sequence(
            torch.cat([
                embedded,
                encodings.unsqueeze(1).expand(embedded.size()[0], embedded.size()[1], encoding_size)
            ], dim = 2),
            lengths.data.cpu().numpy().tolist(),
            batch_first = True
        )

        # Feed packed sequence and encodings
        output, hn = self.gru(packed_sequence)

        output, _ = rnn.pad_packed_sequence(output, batch_first = True)

        seq_len = output.size()[1]

        # Apply output. This has (batch_size) x (seq_length) x (vocab_size)
        predictions = F.log_softmax(self.output(output).view(-1, self.vocab_size)).view(-1, seq_len, self.vocab_size)

        # Select wanted indices
        losses = torch.gather(predictions, 2, targets[:, :seq_len].unsqueeze(2)).squeeze()

        # Mask
        masked_losses = losses * mask[:, :seq_len].float()

        #print(masked_losses)

        # Mean over each sentence and then over cases
        case_loss = masked_losses.sum(1) / lengths.float()

        return -case_loss.mean()

    def signature(self):
        return {
            'type': 'GRUDecoder',
            'dropout': self.dropout,
            'hidden_size': self.hidden_size
        }

class GRUFoldedAverage(nn.Module):
    '''
    GRU 'folded average', which sums
    forward and backward encodings instead of
    concatenating them
    '''
    def __init__(self,
                dropout = 0.3,
                input_size = 202,
                hidden_size = 180):
        super(GRUFoldedAverage, self).__init__()
        self.gru_average = GRUAverage(dropout, input_size, hidden_size, True)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def output_size(self):
        return self.hidden_size

    def signature(self):
        return {
            'type': 'GRUFoldedAverage',
            'gru': self.gru_average.signature()
        }

    def forward(self, batch, padding_mask):
        return F.relu(self.linear(self.gru_average(batch, padding_mask)))

class GRUAverage(nn.Module):
    '''
    Mean-pooled GRU
    '''
    def __init__(self,
                dropout = 0.3,
                input_size = 202,
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
        self.pad_index = input_size - 2

    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def signature(self):
        return {
            'type': 'GRUAverage',
            'dropout': self.dropout,
            'hidden_size': self.hidden_size,
            'bidirectional': self.bidirectional
        }

    def forward(self, batch, padding_mask):
        lengths = padding_mask.long().sum(1)

        lengths, indices = lengths.sort(0, descending = True)

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

class AttentionIsAllYouNeed(nn.Module):
    '''
    Self-attention 'Transformer' network from the 'Attention is All You Need'
    paper.
    '''
    def __init__(self, heads = 3, out_embedding_size = 200,
                key_size = 64, value_size = 64, dropout_value = 0.3):
        super(AttentionIsAllYouNeed, self).__init__()
        self.heads = heads
        self.out_embedding_size = out_embedding_size
        self.key_size = key_size
        self.value_size = value_size

        # The initial transform, which generates
        # all the queries, keys, and values.
        self.initial_transform = nn.Linear(
            202 + 20, # TODO make size of position encoding parameterizable
            self.heads * (2 * key_size + value_size)
        )

        # The merge transform, which takes the multiple
        # attention heads and produces the output encoding
        self.merge_transform = nn.Linear(
            self.heads * value_size,
            self.out_embedding_size
        )

        self.dropout_value = dropout_value
        self.dropout = nn.Dropout(self.dropout_value)

        self.frequency_matrix = Variable(torch.pow(
            1e-4, torch.arange(1, 11).cuda() / 10).view(1, -1),
            requires_grad = False)

    def output_size(self):
        return self.out_embedding_size

    def signature(self):
        return {
            'type': 'AttenionIsAllYouNeed',
            'out_embedding_size': self.out_embedding_size,
            'heads': self.heads,
            'positional_encodings': 10,
            'key_size': self.key_size,
            'value_size': self.value_size,
            'dropout_value': self.dropout_value
        }

    def forward(self, batch, padding_mask):
        # batch will be of size (batch_size) x (seq_length) x (embedding_size)
        # We will perform a self-attention embedding.
        # We want to compute (head) keys, values and queries for each word.

        key_size, value_size, heads = self.key_size, self.value_size, self.heads

        batch_size, seq_length, embedding_size = batch.size()
        pos_size = self.frequency_matrix.size()[1]

        # Positional encoding
        positions = Variable(torch.arange(0, seq_length).cuda().view(
            seq_length, 1), requires_grad = False)
        position_ratios = torch.torch.mm(positions, self.frequency_matrix)
        sin_encoding = torch.sin(position_ratios
            ).unsqueeze(0).expand(batch_size, seq_length, pos_size)
        cos_encoding = torch.cos(position_ratios
            ).unsqueeze(0).expand(batch_size, seq_length, pos_size)
        batch = torch.cat([batch, sin_encoding, cos_encoding], dim = 2)

        # Update embedding size to match embedding with positional encoding
        embedding_size += pos_size * 2

        # Get queries, keys, values
        all_info = F.relu(self.initial_transform(
            batch.view(-1, embedding_size)).view(batch_size, seq_length, -1))

        # Each of these should be (batch_size) x (seq_length) x (heads) x (dim)
        queries = all_info[:, :, :(self.heads * key_size)].contiguous()
        keys = all_info[:, :, (self.heads * key_size):(self.heads * key_size) * 2].contiguous()
        values = all_info[:, :, -(self.heads * value_size):].contiguous()

        queries = queries.view(batch_size, seq_length, heads, key_size)
        keys = queries.view(batch_size, seq_length, heads, key_size)
        values = queries.view(batch_size, seq_length, heads, value_size)

        # Batched matrix multiplication to get query-key similarity.
        # We need to transpose queries and keys to put heads first,
        # so that we can batch them properly.
        queries = queries.transpose(1, 2)
        keys_transpose = queries.transpose(1, 2).transpose(2, 3)
        values_transpose = queries.transpose(1, 2)

        queries = self.dropout(queries)
        keys = self.dropout(keys)
        values = self.dropout(values)

        # Get attentions
        attentions = torch.bmm(
            queries.view(-1, seq_length, key_size),
            keys.view(-1, key_size, seq_length)
        ).view(batch_size, heads, seq_length, seq_length)

        # "Scaled dot-product attention"
        attentions /= math.sqrt(key_size)

        # Softmax over the last dimension
        attentions = F.softmax(
            attentions.view(-1, seq_length)).view(batch_size, heads, seq_length, seq_length)

        # Mask out padding vectors
        attentions = attentions * padding_mask.unsqueeze(1).unsqueeze(1).expand_as(attentions)
        attentions = attentions / attentions.sum(dim = 3).unsqueeze(3).expand_as(attentions)

        # Apply attention to values
        # This should produce a matrix of size (batch_size) x (heads) x (seq_len) x (value_dim)
        aggregations = torch.bmm(
            attentions.view(-1, seq_length, seq_length),
            values.view(-1, seq_length, value_size)
        ).view(batch_size, heads, seq_length, value_size)

        # Re-concatenate all of the aggregations
        aggregations = aggregations.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        # Merge transform, yielding (batch_size) x (seq_len) x (embedding_size)
        embeddings = self.merge_transform(
            aggregations.view(batch_size * seq_length, -1)).view(batch_size, seq_length, -1)

        # Average embedding
        return (
            (embeddings * padding_mask.unsqueeze(2).expand_as(embeddings)).sum(1) /
            padding_mask.sum(1).unsqueeze(1).expand(batch_size, self.out_embedding_size)
        )

class TwoLayerDiscriminator(nn.Module):
    '''
    Simple multi-layer perceptron
    with ReLU nonlinearity.
    '''
    def __init__(self, input_size = 280, hidden_size = 300):
        super(TwoLayerDiscriminator, self).__init__()

        self.first_layer = nn.Linear(input_size, hidden_size)
        self.second_layer = nn.Linear(hidden_size, 2)

    def forward(self, x):
        return F.log_softmax(
            self.second_layer(F.relu(self.first_layer(x)))
        )
