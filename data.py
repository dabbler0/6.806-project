import torch
from torch.utils.data import *

'''
Vocabulary and tokenizers
'''
class Vocabulary:
    def __init__(self, fname):
        self.vocabulary = []
        self.word_to_idx = {}
        self.embedding = []
        with open(fname) as vocab_file:
            for line in vocab_file:
                word = line[:line.index(' ')]
                vector = [int(x) for x in line[line.index(' '):].split(' ')]
                self.word_to_idx[word] = len(self.vocabulary)
                self.vocabulary.append(word)
                self.embedding.append(vector)

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

class QuestionDataset(Dataset):
    def __init__(self, train, questions, vocabulary):
        self.questions = {}
        self.entries = []
        self.vocabulary = vocabulary

        with open(questions) as question_file:
            for line in questisons:
                qid, title, body = line.split('\t')
                qid = int(qid)

                title = [vocabulary.word_to_idx[t] for t in title.split(' ')]
                body = [vocabulary.word_to_idx[t] for t in body.split(' ')]

                self.questions[qid] = Question(qid, title, body)

        with open(train) as train_file:
            for line in train_file:
                q, similar, random = [
                    [self.questions[int(y)] for y in x.split(' ')]
                    for x in line.split('\t')
                ]
                self.entries.append(SimilarityEntry(q[0], similar, random))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]

