import torch
import torch.nn as nn

class CosineSimilarityMaster(nn.Module):
    def __init__(self, word_embedding, sentence_embedding):
        self.word_embedding = word_embedding
        self.sentence_embedding = sentence_embedding

    def forward(self, q, similar, random):
        # At this point the sentences will be encoded as batch_size x truncate_length,
        # batch_size x 20 x truncate_length
        # Get embeddings for all the questions.
        q = self.sentence_embedding(self.word_embedding(q))
        similar = self.sentence_embedding(self.word_embedding(similar))
        random = self.sentence_embedding(self.word_embedding(random.view(-1, truncate_length))).view(-1, 20, truncate_length)

