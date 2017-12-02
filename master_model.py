import torch
import torch.nn as nn

class CosineSimilarityMaster(nn.Module):
    def __init__(self, word_embedding, sentence_embedding, truncate_length, margin):
        self.word_embedding = word_embedding
        self.sentence_embedding = sentence_embedding
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.truncate_length = truncate_length
        self.margin = margin

    def forward(self, q, similar, random):
        truncate_length, margin = self.truncate_length, self.margin

        # At this point the sentences will be encoded as batch_size x truncate_length,
        # random will be batch_size x 20 x truncate_length
        # Get embeddings for all the questions.
        q = self.sentence_embedding(self.word_embedding(q))
        similar = self.sentence_embedding(self.word_embedding(similar))
        random = self.sentence_embedding(self.word_embedding(random.view(-1, truncate_length))).view(-1, 20, truncate_length)

        # Take cosine simliarities
        similar_similarity = self.cosine_similarity(q, similar)
        random_similarity = self.cosine_similarity(q.view(-1, truncate_length, 1).expand_as(random), random)

        # The similarities should be of size batch_size and batch_size x 20
        # Take maximum similarity
        maximum_random_similarity, _indices = torch.max(random_similarity, 1)

        # Add hinge loss
        maximum_random_similarity += margin

        # Get total maximum similarity
        maximum_similarity = torch.max(torch.stack(maximum_random_similarity, similar_similarity))

        # Hinge loss.
        return similar_similarity - maximum_similarity
