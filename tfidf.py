import torch
torch.manual_seed(0)

from master_model import *
from data import *
from simple_models import *

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import json

version = 'third_versioned; batchnorm'

# Hyperparameters
def train(embedder,
            save_dir,
            title_length = 40,
            body_length = 100,
            batch_size = 10,
            test_batch_size = 10,
            margin = 0.1,
            epochs = 50,
            lr = 1e-4,
            cuda = True,
            body_embedder = None,
            merge_strategy = 'mean',
            negative_samples = 20,
            output_embedding_size = 0,
            alpha = 1e-4,
            vectors = 'askubuntu/vector/vectors_pruned.200.txt',
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


    # print type(mapping)
    # print mapping[0].shape
    # print vectorizer.get_feature_names()[:30]
    # print len(vectorizer.get_feature_names())








train(
    embedder = CNN(), #False),
    save_dir='models/cnn',
    batch_size = 200,
    test_batch_size = 200,
    lr = 1e-4,

    title_length = 40,
    negative_samples = 20,
    alpha = 0,

    body_embedder = CNN(),
    body_length = 50,
    merge_strategy = 'mean',
    output_embedding_size = 400,

    epochs = 50,
    margin = 0.2
)
