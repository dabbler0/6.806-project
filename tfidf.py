import torch
torch.manual_seed(0)

from master_model import *
from data import *
from architectures import *

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import json

# Hyperparameters
def train_tfidf(
            android_questions_filename = 'android/corpus.tsv',

            dev_pos_txt = 'android/dev.pos.txt',
            dev_neg_txt = 'android/dev.neg.txt',
            test_neg_txt = 'android/test.neg.txt',
            test_pos_txt = 'android/test.pos.txt'):

    mapping = {}
    for i, line in enumerate(open(android_questions_filename)):
        q, t, b = line.split('\t')
        mapping[int(q)] = i

    corpus = open(android_questions_filename)
    vectorizer = TfidfVectorizer()
    rep = vectorizer.fit_transform(corpus)

    tester = TFIDFTestFramework((dev_pos_txt, dev_neg_txt), mapping, rep)
    print tester.metrics()

if __name__ == '__main__':
    train_tfidf()
