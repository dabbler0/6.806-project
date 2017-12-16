import torch
from torch.utils.data import *
from tqdm import tqdm
import numpy as np
import random



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
        self.questions = {-1: ([0] * title_length, [0] * body_length)}
        self.vocabulary = vocabulary
        self.title_length = title_length
        self.body_length = body_length

        with open(questions) as question_file:
            for line in tqdm(question_file, desc='load questions'):
                if len(line.split('\t')) != 3:
                    print "error", line
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

    def randomKey(self):
        entry = -1
        while entry == -1:
            entry = random.choice(self.questions.keys())
        return entry

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


class AndroidTrainSet(Dataset):
    def __init__(self, train, ubuntu_questions, android_questions):
        self.ubuntu_questions = ubuntu_questions
        self.android_questions = android_questions
        self.entries = []



        ubuntu_entry = ubuntu_questions.randomKey()

        android_entry = android_questions.randomKey()

        #ubuntu_entry, android_entry = randint(2, len(self.ubuntu_questions)-1), randint(2, len(self.android_questions)-1)

        ubuntu_title_entry, android_title_entry = self.ubuntu_questions[ubuntu_entry][0], self.android_questions[android_entry][0]
        ubuntu_body_entry, android_body_entry = self.ubuntu_questions[ubuntu_entry][1], self.android_questions[android_entry][1]

        #entries are 0 if from ubuntu, and 1 if from android
        self.entries.append({
            'title' : torch.stack([torch.LongTensor(ubuntu_title_entry), torch.LongTensor(android_title_entry)]),
            'body' : torch.stack([torch.LongTensor(ubuntu_body_entry), torch.LongTensor(android_body_entry)]),
            'label': torch.FloatTensor([0, 1])
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


class AndroidTestSet:
    def __init__(self, test_set, padding=True):
        self.entries = []
        self.pos_set, self.neg_set = test_set

        neg_count = 0
        previous_q = ''
        previous_question = ''
        current = ''
        prev_random = []
        neg_added = 0
        with open(self.neg_set) as test_file:
            for line in tqdm(test_file, desc='load testset'):
                curr_q, curr_random = line.split(' ')
                if curr_q == previous_q:
                    prev_random.append(int(curr_random))

                else:
                    if neg_count != 0:
                        added = len(prev_random)
                        if padding:
                            prev_random.extend([-1 for _ in range(300-len(prev_random))])
                        self.entries.append({
                            'full_mask': added,
                            'current': previous_q,
                            'q': prev_question,
                            'full': prev_random
                        })
                    prev_question, prev_random = int(curr_q), [int(curr_random)]

                previous_q = curr_q
                neg_count += 1

        added = len(prev_random)
        if padding:
            prev_random.extend([-1 for _ in range(300-len(prev_random))])

        self.entries.append({
            'full_mask': added,
            'current': previous_q,
            'q': prev_question,
            'full': prev_random
        })

        pos_count = 0
        previous_q = ''
        prev_similar = []
        added = 0
        with open(self.pos_set) as test_file:
            for line in tqdm(test_file, desc='load testset'):
                curr_q, curr_similar = line.split(' ')
                if curr_q == previous_q:
                    prev_similar.append(int(curr_similar))

                else:
                    if pos_count != 0:
                        previous_length = len(prev_similar)
                        if padding:
                            prev_similar.extend([-1 for _ in range(3-len(prev_similar))])
                        entry = self.entries[added]
                        entry['similar'] = prev_similar
                        entry['similar_mask'] = previous_length
                        added += 1
                    prev_similar = [int(curr_similar)]

                previous_q = curr_q
                pos_count += 1

        previous_length = len(prev_similar)
        if padding:
            prev_similar.extend([-1 for _ in range(3-len(prev_similar))])
        entry = self.entries[added]
        entry['similar'] = prev_similar
        entry['similar_mask'] = previous_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]
