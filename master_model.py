import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data import *

# Regularization parameter for how much we want to penalize everything converging
alpha = 0.1

class FullEmbedder(nn.Module):
    def __init__(self, vocabulary, sentence_embedding):
        super(FullEmbedder, self).__init__()

        # Word embedding
        self.word_embedding = nn.Embedding(vocabulary.embedding.size()[0], vocabulary.embedding.size()[1])
        self.word_embedding.weight.data = vocabulary.embedding
        self.word_embedding.weight.requires_grad = False

        # Sentence embedding module
        self.sentence_embedding = sentence_embedding

    def forward(self, batch):
        return self.sentence_embedding(self.word_embedding(batch))

class CosineSimilarityMaster(nn.Module):
    def __init__(self, full_embedder, truncate_length, margin):
        super(CosineSimilarityMaster, self).__init__()

        self.embedder = full_embedder

        # Cosine simliarty module
        self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

        # Stored hyperamareters
        self.truncate_length = truncate_length
        self.margin = margin

    def forward(self, q, similar, random):
        truncate_length, margin = self.truncate_length, self.margin

        # At this point the sentences will be encoded as batch_size x truncate_length,
        # random will be batch_size x 20 x truncate_length
        # Get embeddings for all the questions.
        q = self.embedder(q)
        similar = self.embedder(similar)

        #print(q.data[0])
        #print(similar.data[0])

        # Flatten random to include 20 * (batch size) sentences,
        # transform with those 20n sentences, then reshape again to be (batch_size) x (20) x (embedding_size)
        random = self.embedder(random.view(-1, truncate_length)).view(-1, 20, q.size()[1])

        # Take cosine simliarities
        similar_similarity = self.cosine_similarity(q.unsqueeze(1), similar.unsqueeze(1)).squeeze()
        random_similarity = self.cosine_similarity(q.unsqueeze(1).expand_as(random), random)

        # The similarities should be of size batch_size and batch_size x 20
        # Take maximum similarity
        maximum_random_similarity, _indices = torch.max(random_similarity, 1)
        #maximum_random_similarity = random_similarity.mean(1)

        # Add hinge loss
        maximum_random_similarity += margin

        # Hinge loss.
        # If we try to minimize this, then we will be trying to minimize
        # maximum_random_similarity and maximize similar_similarity,
        # which is our intention.
        return F.relu(maximum_random_similarity - similar_similarity)

class TestFramework:
    def __init__(self, test, questions, truncate_length, cuda = True):
        self.test_set = TestSet(test, questions)
        self.truncate_length = truncate_length

        self.cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

        # LongTensor of (cases) x (trunc_length)
        self.question_vector = torch.LongTensor([
            self.test_set.questions[x['q']] for x in self.test_set.entries
        ])

        # LongTensor of (cases) x (num_full) x (trunc_length)
        self.full_vector = torch.LongTensor([
            [self.test_set.questions[y] for y in x['full']]
            for x in self.test_set.entries
        ])

        self.similar_sets = [
            set(x['full'].index(i) for i in x['similar'] if i in x['full'])
            for x in self.test_set.entries
        ]

        # if cuda:
        #     self.question_vector = self.question_vector.cuda()
        #     self.full_vector = self.full_vector.cuda()

    def metrics(self, embedder):
        # Get embeddings for all the questions
        # (cases) x (sent_embedding_size)
        question_embedding = embedder(self.question_vector)

        full_size = self.full_vector.size()

        # Want (cases) x (num_full) x (sent_embedding_size)
        full_embedding = embedder(
            self.full_vector.view(-1, self.truncate_length)
        ).view(full_size[0], full_size[1], -1)

        # Get cosine similarities
        similarities = self.cos_similarity(
            question_embedding.unsqueeze(1).expand_as(full_embedding),
            full_embedding
        )

        # Now we have (cases) x (num_full) different similarities.
        # We want to iterate over these in sorted order for each case.
        sorted_similarities, indices = similarities.sort(dim = 1, descending = True)

        # Now, just in interpreted Python, determine MAP.
        mean_average_precision = 0.0
        mean_reciprocal_rank = 0.0
        precision_at_5n = np.zeros(5)

        samples = 0
        total_samples = 0
        for i, case in enumerate(indices):
            avg_precision = 0.0
            num_recalls = 0.0

            correct_so_far = 0.0
            total_so_far = 0.0

            reciprocal_rank = 0.0
            precision_at_case_5n = np.zeros(5)

            first_examined = False
            for j, candidate in enumerate(case):
                candidate = candidate.data[0]
                total_so_far += 1
                if candidate in self.similar_sets[i]:
                    # For each new possible recall, get precision
                    correct_so_far += 1
                    avg_precision += (correct_so_far / total_so_far)
                    if not first_examined:
                        reciprocal_rank = 1/float(total_so_far)
                        first_examined = True
                    num_recalls += 1

                if j < 5:
                    precision_at_case_5n[j] = num_recalls/total_so_far


            if num_recalls > 0:
                avg_precision /= num_recalls
                mean_average_precision += avg_precision
                mean_reciprocal_rank += reciprocal_rank
                precision_at_5n = np.sum((precision_at_5n, precision_at_case_5n), axis=0)
                samples += 1

        return (mean_average_precision / samples, mean_reciprocal_rank / samples, precision_at_5n / samples)
