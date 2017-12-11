import torch
from torch.utils.data import *
from tqdm import tqdm

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
                vector.append(1) # This extra dimension indicates that this is not a padding vector.
                vector.append(0) # This extra dimension indicates that this is not an unknown vector.
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
    def __init__(self, questions, vocabulary, truncate_length):
        self.questions = {}
        self.vocabulary = vocabulary
        self.truncate_length = truncate_length

        with open(questions) as question_file:
            for line in tqdm(question_file, desc='load questions'):
                qid, title, body = line.split('\t')
                qid = int(qid)

                title = [vocabulary.to_idx(t) for t in title.split(' ')]

                #body = [vocabulary.to_idx(t) for t in body.split(' ')]
                # TODO working only with title for now

                if len(title) <= truncate_length:
                    # Pad the title
                    title += [0] * (truncate_length - len(title))
                    self.questions[qid] = title
                else:
                    self.questions[qid] = title[:truncate_length]

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
                    'q': torch.LongTensor(q[0]),
                    'similar': torch.LongTensor(similar[0]),
                    'random': torch.LongTensor(random[:20]) # Make it size twenty
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
                    #Question
                    [int(y) for y in x.split(' ') if len(y) > 0]
                    for x in line.split('\t')[:-1]
                ]
                self.entries.append({
                    'q': q[0],
                    'similar': similar,
                    'full': full
                })
