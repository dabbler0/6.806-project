import torch
torch.manual_seed(0)

from master_model import *
from data import *
from simple_models import *

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

import os
import json

def test(filename,
         before_domain=True,
         dev_pos_txt = 'android/dev.pos.txt',
         dev_neg_txt = 'android/dev.neg.txt',
         title_length = 40,
         body_length = 100,
         batch_size = 100,
         test_batch_size = 100,
         vectors = 'askubuntu/vector/vectors_pruned.200.txt',
         ubuntu_questions_filename = 'askubuntu/text_tokenized.txt',
         test_set = 'askubuntu/test.txt'):

    full_embedder = torch.load(filename)
    vocabulary = Vocabulary(vectors, [ubuntu_questions_filename])

    ubuntu_questions = QuestionBase(ubuntu_questions_filename, vocabulary, title_length, body_length)
    #android_questions = QuestionBase(android_questions_filename, vocabulary, title_length, body_length)

    tester = TestFramework(test_set, ubuntu_questions, title_length, body_length)
    #android_tester = AndroidTestFramework((dev_pos_txt, dev_neg_txt), android_questions, title_length, body_length, test_batch_size, num_examples = 100)

    mean_average_precision, mean_reciprocal_rank, precision_at_n = tester.metrics(full_embedder)

    print('test MAP = %f, test MRR = %f \n, precision@1 = %f, precision@5 = %f' % (mean_average_precision, mean_reciprocal_rank, precision_at_n[0], precision_at_n[4]))

def test_transfer(filename,
         before_domain = True,
         dev_pos_txt = 'android/test.pos.txt',
         dev_neg_txt = 'android/test.neg.txt',
         title_length = 40,
         body_length = 100,
         test_batch_size = 10,
         vectors = 'glove/glove.840B.300d.txt',
         ubuntu_questions_filename = 'askubuntu/text_tokenized.txt',
         android_questions_filename = 'Android/corpus.tsv',
         test_set = 'askubuntu/test.txt'):

    full_embedder = torch.load(filename)
    if type(full_embedder) is tuple:
        _, full_embedder = full_embedder
    vocabulary = Vocabulary(vectors, [ubuntu_questions_filename, android_questions_filename], android_questions_filename)

    android_questions = QuestionBase(android_questions_filename, vocabulary, title_length, body_length)

    tester = AndroidTestFramework((dev_pos_txt, dev_neg_txt), android_questions, title_length, body_length, test_batch_size, num_examples = 100)

    auc = tester.metrics(full_embedder)
    print('AUC = %f' % auc)

def test_unsupervised(filename,
         before_domain = True,
         dev_pos_txt = 'android/dev.pos.txt',
         dev_neg_txt = 'android/dev.neg.txt',
         title_length = 40,
         body_length = 100,
         test_batch_size = 10,
         vectors = 'glove/glove.840B.300d.txt',
         ubuntu_questions_filename = 'askubuntu/text_tokenized.txt',
         android_questions_filename = 'Android/corpus.tsv',
         test_set = 'askubuntu/test.txt'):

    full_embedder = torch.load(filename)
    vocabulary = Vocabulary(vectors, [android_questions_filename], android_questions_filename)

    android_questions = QuestionBase(android_questions_filename, vocabulary, title_length, body_length)

    tester = AndroidTestFramework((dev_pos_txt, dev_neg_txt), android_questions, title_length, body_length, test_batch_size, num_examples = 100)

    auc = tester.metrics(full_embedder)
    print('AUC = %f' % auc)


#test('models/cnn-1/best.pkl')
#test_transfer('models/gru-direct-transfer-fixed/best.pkl')
#test_transfer('models/gru-direct-transfer-fixed/epoch13-loss0.585714.pkl')
test_transfer('models/domain-adaptation-gru-best/epoch13-loss0.645168.pkl')
