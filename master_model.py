'''
"Master" wrapper models for all of the models defined in architectures.py.

A model in architectures.py defines a transformation from word embedding vectors
into a question embedding. The models in this file wrap those models, transforming
word indices into word embeddings to feed into them and then merging title and body
embeddings, doing BatchNorm, and doing cosine simliarity for testing.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data import *
from meter import *
from scipy.sparse import csr_matrix, find

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class FullEmbedder(nn.Module):
    '''
    FullEmbedder takes a word embedding vocabulary, title and body embedding, and is
    an embedder for (title, body) tuples. It contains a BatchNorm layer,
    and depending on what merge_strategy is, possibly also a linear layer
    used to combine title and body embeddings.
    '''
    def __init__(self,
                    vocabulary,
                    title_embedding,
                    body_embedding,
                    merge_strategy = 'mean',
                    output_embedding_size = 0,
                    train_embeddings = False):

        super(FullEmbedder, self).__init__()

        # Word embedding
        self.word_embedding = nn.Embedding(vocabulary.embedding.size()[0], vocabulary.embedding.size()[1])
        self.word_embedding.weight.data = vocabulary.embedding

        self.word_embedding.weight.requires_grad = train_embeddings

        # Sentence embedding module
        self.title_embedding = title_embedding
        self.body_embedding = body_embedding

        self.merge_strategy = merge_strategy

        # Determine output size if not given (it is only
        # given when merge_strategy is 'linear')
        total_size = title_embedding.output_size()
        if body_embedding is not None:
            total_size += body_embedding.output_size()

        # Initialize desired merge strategy
        if self.merge_strategy == 'linear':
            self.merge_linear = nn.Linear(total_size, output_embedding_size, bias = False)
        elif self.merge_strategy == 'concatenate':
            output_embedding_size = total_size
        elif self.merge_strategy == 'mean':
            output_embedding_size = title_embedding.output_size()
        else:
            raise Exception('Unrecognized merge strategy \'%s\'' % (merge_strategy,))

        self.batch_norm = nn.BatchNorm1d(output_embedding_size)

    def forward(self, pair):
        title, body = pair

        if self.body_embedding is not None:
            # Run both title and body encodings
            title_encodings = self.title_embedding(self.word_embedding(title), title.gt(0))
            body_encodings = self.body_embedding(self.word_embedding(body), body.gt(0))

            # Apply desired merge strategy
            if self.merge_strategy == 'concatenate':
                result = torch.cat([title_encodings, body_encodings], dim = 1)
            elif self.merge_strategy == 'linear':
                result = self.merge_linear(torch.cat([title_encodings, body_encodings], dim = 1))
            elif self.merge_strategy == 'mean':
                result = (title_encodings + body_encodings) / 2

        # FullEmbedder also works if there is
        # no body embedding, in which case the title embedding
        # is used alone
        else:
            result = self.title_embedding(self.word_embedding(title))

        # BatchNorm
        return self.batch_norm(result)

    def regularizer(self):
        # Using 'linear' merge strategy requires
        # very strong regularization to encourage zero-weights,
        # otherwise it never trains.

        # If 'linear' is used then this function needs to be called
        # from the training function and used as a regularizer.
        if self.merge_strategy == 'linear':
            return self.merge_linear.weight.norm(0.5)
        return 0

class BodyOnlyEmbedder(nn.Module):
    '''
    A FullEmbedder-like class that only runs an embedder on the body.
    This is used by the Body-to-Title summarization model.
    '''
    def __init__(self, vocabulary, body_embedding, train_embeddings = False):
        super(BodyOnlyEmbedder, self).__init__()

        # Word embedding
        self.word_embedding = nn.Embedding(vocabulary.embedding.size()[0], vocabulary.embedding.size()[1])
        self.word_embedding.weight.data = vocabulary.embedding
        self.word_embedding.weight.requires_grad = train_embeddings

        # Body embedding
        self.body_embedding = body_embedding

        # BatchNorm
        self.batch_norm = nn.BatchNorm1d(body_embedding.output_size())

    def forward(self, pair):
        title, body = pair
        return self.batch_norm(self.body_embedding(self.word_embedding(body), body.gt(0)))

class CosineSimilarityMaster(nn.Module):
    '''
    CosineSimilarity is a framework which takes a full_embedder
    and can be run on batches with a single similar question
    and multiple random questions and return the max margin loss
    obtained with the given embedder and batch.
    '''
    def __init__(self, full_embedder, title_length, body_length, margin):
        super(CosineSimilarityMaster, self).__init__()

        self.embedder = full_embedder

        # Cosine simliarty module
        self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

        # Stored hyperamareters
        self.title_length = title_length
        self.body_length = body_length
        self.margin = margin

    def forward(self, q, similar, random):
        title_length, body_length, margin = self.title_length, self.body_length, self.margin

        # At this point the sentences will be encoded as batch_size x length
        # random will be batch_size x 20 x length
        # Get embeddings for all the questions.
        q = self.embedder(q)
        similar = self.embedder(similar)

        # To embed all random sentences at once,
        # flatten random to include 20 * (batch size) sentences,
        # transform with those 20n sentences,
        # then reshape again to be (batch_size) x (20) x (embedding_size)
        negative_samples = random[0].size()[1]
        random = self.embedder(
            (random[0].view(-1, title_length),
             random[1].view(-1, body_length))
        ).view(-1, negative_samples, q.size()[1])

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
    '''
    Framework for computing MAP, MRR, etc.
    '''
    def __init__(self, test, questions, title_length, body_length, cuda = True):
        self.test_set = TestSet(test, questions)

        self.title_length = title_length
        self.body_length = body_length

        self.cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

        # LongTensor of (cases) x (trunc_length)
        self.question_title_vector = Variable(torch.LongTensor([
            self.test_set.questions[x['q']][0] for x in self.test_set.entries
        ]))
        self.question_body_vector = Variable(torch.LongTensor([
            self.test_set.questions[x['q']][1] for x in self.test_set.entries
        ]))

        # LongTensor of (cases) x (num_full) x (trunc_length)
        self.full_title_vector = Variable(torch.LongTensor([
            [self.test_set.questions[y][0] for y in x['full']]
            for x in self.test_set.entries
        ]))

        self.full_body_vector = Variable(torch.LongTensor([
            [self.test_set.questions[y][1] for y in x['full']]
            for x in self.test_set.entries
        ]))

        self.similar_sets = [
            set(x['full'].index(i) for i in x['similar'] if i in x['full'])
            for x in self.test_set.entries
        ]

        if cuda:
            self.question_title_vector = self.question_title_vector.cuda()
            self.full_title_vector = self.full_title_vector.cuda()

            self.question_body_vector = self.question_body_vector.cuda()
            self.full_body_vector = self.full_body_vector.cuda()

        self.question_vector = (self.question_title_vector, self.question_body_vector)

    def metrics(self, embedder):
        # Get embeddings for all the questions
        # (cases) x (sent_embedding_size)
        question_embedding = embedder(self.question_vector)

        full_size = self.full_title_vector.size()

        # Want (cases) x (num_full) x (sent_embedding_size)
        full_embedding = embedder(
            (self.full_title_vector.view(-1, self.title_length),
            self.full_body_vector.view(-1, self.body_length))
        ).view(full_size[0], full_size[1], -1)

        # Get cosine similarities
        similarities = self.cos_similarity(
            question_embedding.unsqueeze(1).expand_as(full_embedding),
            full_embedding
        )

        # Now we have (cases) x (num_full) different similarities.
        # We want to iterate over these in sorted order for each case.
        sorted_similarities, indices = similarities.sort(dim = 1, descending = True)

        # Now, just in interpreted Python, determine metrics.
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


    def visualize_embeddings(self, embedder, filename):
        question_embedding = embedder(self.question_vector)
        plt.matshow(question_embedding.data.cpu().numpy())
        plt.savefig(filename)

class AndroidTestFramework:
    '''
    Framework for determining AUC using annotations like in the Android corpus.
    '''
    def __init__(self, test, questions, title_length, body_length, test_batch_size, num_examples = None, cuda = True):
        self.test_set = AndroidTestSet(test, questions, num_examples)
        self.batch_size = test_batch_size
        self.test_loader = DataLoader(
                AndroidTestSet(test, questions),
                batch_size = self.batch_size,
                shuffle = False,
                drop_last = False
        )
        self.title_length = title_length
        self.body_length = body_length

        self.cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.AUC = AUCMeter()

        self.visualize_batch = self.test_set.questions.get_random_batch(200)

    def metrics(self, embedder):
        # Get embeddings for all the questions
        # (cases) x (sent_embedding_size)
        self.AUC.reset()
        for i, batch in enumerate(tqdm(self.test_loader)):

            self.question_title_vector = Variable(torch.LongTensor([
                self.test_set.questions[x][0] for x in batch['q']
            ]).cuda())

            self.question_body_vector = Variable(torch.LongTensor([
                self.test_set.questions[x][1] for x in batch['q']
            ]).cuda())

            # LongTensor of (cases) x (num_full) x (trunc_length)
            self.full_title_vector = Variable(torch.LongTensor([
                [self.test_set.questions[y][0] for y in x]
                for x in batch['full']
            ]).cuda().transpose(0, 1).contiguous())


            self.full_body_vector = Variable(torch.LongTensor([
                [self.test_set.questions[y][1] for y in x]
                for x in batch['full']
            ]).cuda().transpose(0, 1).contiguous())

            # LongTensor of (cases) x (num_full) x (trunc_length)
            self.similar_title_vector = Variable(torch.LongTensor([
                [self.test_set.questions[y][0] for y in x]
                for x in batch['similar']
            ]).cuda().transpose(0, 1).contiguous())


            self.similar_body_vector = Variable(torch.LongTensor([
                [self.test_set.questions[y][1] for y in x]
                for x in batch['similar']
            ]).cuda().transpose(0, 1).contiguous())

            self.question_vector = (self.question_title_vector, self.question_body_vector)

            self.similar_mask = torch.IntTensor([int(x) for x in batch['similar_mask']])
            self.full_mask = torch.IntTensor([int(x) for x in batch['full_mask']])

            question_embedding = embedder(self.question_vector)

            full_size = self.full_title_vector.size()

            # Want (cases) x (num_full) x (sent_embedding_size)
            full_embedding = embedder(
                (self.full_title_vector.view(-1, self.title_length),
                self.full_body_vector.view(-1, self.body_length))
            ).view(full_size[0], full_size[1], -1)

            similar_size = self.similar_title_vector.size()
            similar_embedding = embedder(
                (self.similar_title_vector.view(-1, self.title_length),
                self.similar_body_vector.view(-1, self.body_length))
            ).view(similar_size[0], similar_size[1], -1)


            # Get cosine similarities
            full_similarities = self.cos_similarity(
                question_embedding.unsqueeze(1).expand_as(full_embedding),
                full_embedding
            ).data.cpu().numpy()

            similar_similarities = self.cos_similarity(
                question_embedding.unsqueeze(1).expand_as(similar_embedding),
                similar_embedding
            ).data.cpu().numpy()


            for i in range(len(full_similarities)):
                curr_mask, curr_sim_mask = self.full_mask[i], self.similar_mask[i]
                comb_similarities = torch.FloatTensor(
                    np.concatenate((full_similarities[i][:curr_mask], similar_similarities[i][:curr_sim_mask]))
                )
                comb_target = torch.FloatTensor(
                    np.concatenate(([0 for _ in range(curr_mask)], [1 for _ in range(curr_sim_mask)]))
                )
                self.AUC.add(comb_similarities, comb_target)

        return self.AUC.value(.05)

    def visualize_embeddings(self, embedder, filename):
        question_embedding = embedder(self.visualize_batch)
        plt.matshow(question_embedding.data.cpu().numpy())
        plt.savefig(filename)

    def sample(self, embedder):
        return embedder(self.question_vector)

class TFIDFTestFramework:
    '''
    Framework for testing TFIDF or other CountVectorizer-like
    models.
    '''
    def __init__(self, test, mapping, rep):
        self.test_set = AndroidTestSet(test, False)

        self.mapping = mapping
        self.rep = rep

        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.AUC = AUCMeter()

    def metrics(self):
        self.AUC.reset()

        for i, entry in enumerate(self.test_set):
            if i % 10 == 0:
                print i

            combined = entry['full'][:entry['full_mask']] + entry['similar'][:entry['similar_mask']]

            row, col, values = find(self.rep[self.mapping[entry['q']]])

            combined_tensor = torch.stack(
                (torch.IntTensor(row), torch.IntTensor(col)),
                dim=0
            ).long()

            target_review = torch.sparse.FloatTensor(
                    combined_tensor,
                    torch.FloatTensor(values),
                    torch.Size([1,79066])
            ).to_dense()

            longing = np.zeros(len(combined))
            for i, review in enumerate(combined):
                frow, fcol, fvalues = find(self.rep[self.mapping[review]])
                full_tensor = torch.stack(
                        (torch.IntTensor(frow), torch.IntTensor(fcol)), dim=0).long()
                full_review = torch.sparse.FloatTensor(full_tensor,
                        torch.FloatTensor(fvalues), torch.Size([1,79066])).to_dense()
                valuation = self.cos_similarity(target_review, full_review).numpy()
                longing[i] = valuation

            comb_similarities = torch.FloatTensor(longing)

            comb_target = torch.FloatTensor(
                np.concatenate(([0 for _ in range(entry['full_mask'])],
                    [1 for _ in range(entry['similar_mask'])]))
            )

            self.AUC.add(comb_similarities, comb_target)

        return self.AUC.value(.05)
