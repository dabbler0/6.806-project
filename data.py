import torch
from torch.utils.data import *
from tqdm import tqdm

word_embedding_size = 202

'''
Vocabulary and tokenizers
'''
class Vocabulary:
    def __init__(self, fname):
        # Token 0 is the padding token.
        # Token 1 is the unknown token.
        self.vocabulary = ['__EOS__', '__UNK__']
        self.word_to_idx = {'__EOS__': 0, '__UNK__': 1}
        self.embedding = [[0] * 202, [0] * 200 + [1, 1]]

        with open(fname) as vocab_file:
            for line in tqdm(vocab_file, desc='load vocab'):
                word = line[:line.index(' ')]
                vector = [float(x) for x in line[line.index(' '):].split(' ')[1:-1]]

                # This is not a padding vector or UNK
                vector += [1, 0]

                self.word_to_idx[word] = len(self.vocabulary)
                self.vocabulary.append(word)
                self.embedding.append(vector)

        self.embedding = torch.Tensor(self.embedding)

    def to_idx(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        return 1

'''
Question datasets
'''

class SimilarityEntry:
    def __init__(self, q, similar, random):
        self.q = q
        self.similar = similar
        self.random = random

class Question:
    def __init__(self, qid, header, body):
        self.qid = qid
        self.header = header
        self.body = body

class QuestionBase:
    def __init__(self, questions, vocabulary, title_length, body_length):
        self.questions = {}
        self.vocabulary = vocabulary
        self.title_length = title_length
        self.body_length = body_length

        with open(questions) as question_file:
            for line in tqdm(question_file, desc='load questions'):
                qid, title, body = line.split('\t')
                qid = int(qid)


                # Add dimension that flags whether we are in the title or the body
                title = [vocabulary.to_idx(t) for t in title.split(' ')]
                if len(title) <= title_length:
                    title += [0] * (title_length - len(title))
                title = title[:title_length]

                body = [vocabulary.to_idx(t) for t in body.split(' ')]
                if len(body) <= body_length:
                    body += [0] * (body_length - len(body))
                body = body[:body_length]

                self.questions[qid] = (title, body)

    def __getitem__(self, item):
        return self.questions[item]

class TrainSet(Dataset):
    def __init__(self, train, questions):
        self.questions = questions
        self.entries = []

        with open(train) as train_file:
            for line in tqdm(train_file, desc='load trainset'):
                q, similar, random = [
                    [self.questions[int(y)] for y in x.split(' ')]
                    for x in line.split('\t')
                ]
                self.entries.append({
                    'q_title': torch.LongTensor(q[0][0]),
                    'q_body': torch.LongTensor(q[0][1]),

                    'similar_title': torch.LongTensor(similar[0][0]),
                    'similar_body': torch.LongTensor(similar[0][1]),

                    # We will arbitrarily use only 20 of the negative samples
                    'random_title': torch.LongTensor([x[0] for x in random]),
                    'random_body': torch.LongTensor([x[1] for x in random])
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]

'''
Test frameworks
'''

class TestSet:
    def __init__(self, test, questions):
        self.questions = questions
        self.entries = []

        with open(test) as test_file:
            for line in tqdm(test_file, desc='load testset'):
                q, similar, full = [
                    [int(y) for y in x.split(' ') if len(y) > 0]
                    for x in line.split('\t')[:-1]
                ]
                self.entries.append({
                    'q': q[0],
                    'similar': similar,
                    'full': full
                })
