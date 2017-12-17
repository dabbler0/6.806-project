import torch
from torch.utils.data import *
from tqdm import tqdm

from torch.autograd import Variable

word_embedding_size = 202

'''
Vocabulary and tokenizers
'''
class Vocabulary:
    def __init__(self, fname, prune_corpora, unprune_corpus, additional_words = 500):
        # Token 0 is the padding token.
        # Token 1 is the unknown token.
        self.vocabulary = ['__EOS__', '__UNK__']
        self.word_to_idx = {'__EOS__': 0, '__UNK__': 1}
        self.embedding = []

        # Record which words appear in the corpora so that
        # we can only include words that actually appear
        prune_candidates = set()
        for corpus_fname in prune_corpora:
            with open(corpus_fname) as corpus:
                for line in corpus:
                    qid, title, body = line.split('\t')
                    words = [x.lower().strip() for x in title.split(' ') + body.split(' ')]
                    prune_candidates.update(words)

        frequencies = {}
        with open(unprune_corpus) as corpus:
            for line in corpus:
                qid, title, body = line.split('\t')
                words = [x.lower().strip() for x in title.split(' ') + body.split(' ')]
                for word in words:
                    if word not in frequencies:
                        frequencies[word] = 0
                    frequencies[word] += 1

        most_common_words = sorted(frequencies, key = lambda k: -frequencies[k])

        with open(fname) as vocab_file:
            for line in tqdm(vocab_file, desc='load vocab'):
                word = line[:line.index(' ')].lower()

                # Only include words that appear in one of the corpora (if any exist)
                if word in prune_candidates or len(prune_corpora) == 0:
                    vector = [float(x) for x in line[line.index(' '):].split(' ')[1:] if len(x) > 0]

                    # Add padding and UNK tokens
                    if len(self.embedding) == 0:
                        self.embedding.append([0] * (len(vector) + 2))
                        self.embedding.append([0] * (len(vector)) + [1, 1])

                    # This is not a padding vector or UNK
                    vector += [1, 0]

                    self.word_to_idx[word] = len(self.vocabulary)
                    self.vocabulary.append(word)
                    self.embedding.append(vector)

        number_added = 0
        for word in most_common_words:
            if word not in self.word_to_idx:
                print('Adding word', word, frequencies[word])
                # Add this word as a random vector
                self.word_to_idx[word] = len(self.vocabulary)
                self.vocabulary.append(word)
                self.embedding.append(
                    torch.Tensor(len(self.embedding[0]) - 2).uniform_().numpy().tolist()
                    + [1, 0]
                )

                number_added += 1
                if number_added >= additional_words:
                    break

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
        self.questions = {-1: ([1] + [0] * (title_length - 1), [1] + [0] * (body_length - 1))}
        self.vocabulary = vocabulary
        self.title_length = title_length
        self.body_length = body_length

        self.all_qids = []

        with open(questions) as question_file:
            for line in tqdm(question_file, desc='load questions'):
                if len(line.split('\t')) != 3:
                    print "error", line
                qid, title, body = line.split('\t')
                qid = int(qid)

                self.all_qids.append(qid)

                # Add dimension that flags whether we are in the title or the body
                title = [vocabulary.to_idx(t.lower().strip()) for t in title.split(' ')]
                if len(title) <= title_length:
                    title += [0] * (title_length - len(title))
                title = title[:title_length]

                body = [vocabulary.to_idx(t.lower().strip()) for t in body.split(' ')]
                if len(body) <= body_length:
                    body += [0] * (body_length - len(body))
                body = body[:body_length]

                self.questions[qid] = (title, body)


        # Make a LongTensor of all possible qids, for random
        # selection
        self.all_qids = torch.LongTensor(self.all_qids)

    def get_random_batch_and_qids(self, batch_size, cuda = True):
        # Get batch_size random indices into the qids tensor
        indices = torch.floor(torch.Tensor(batch_size).uniform_() * self.all_qids.size()[0]).long()

        # Get batch_size random qids
        qids = self.all_qids[indices]

        title_list = []
        body_list = []

        for qid in qids:
            title, body = self[qid]
            title_list.append(title)
            body_list.append(body)

        # Make into tensors for processing
        title_tensor = torch.LongTensor(title_list)
        body_tensor = torch.LongTensor(body_list)

        if cuda:
            title_tensor = title_tensor.cuda()
            body_tensor = body_tensor.cuda()

        return qids, (Variable(title_tensor), Variable(body_tensor))

    def get_random_batch(self, batch_size, cuda = True):
        return self.get_random_batch_and_qids(batch_size, cuda)[1]

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
    def __init__(self, test_set, questions, num_examples = None):
        self.questions = questions
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
                        prev_similar.extend([-1 for _ in range(3-len(prev_similar))])
                        entry = self.entries[added]
                        entry['similar'] = prev_similar
                        entry['similar_mask'] = previous_length
                        added += 1
                    prev_similar = [int(curr_similar)]

                previous_q = curr_q
                pos_count += 1

        previous_length = len(prev_similar)
        prev_similar.extend([-1 for _ in range(3-len(prev_similar))])
        entry = self.entries[added]
        entry['similar'] = prev_similar
        entry['similar_mask'] = previous_length

        if num_examples is not None:
            self.entries = self.entries[:num_examples]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        return self.entries[i]
