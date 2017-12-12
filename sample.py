import torch
torch.manual_seed(0)

from master_model import *
from data import *
from simple_models import *

import math

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

import os
import json

import numpy as np
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

version = 'third_versioned; batchnorm'

def sample(full_embedder,
            output,
            title_length = 40,
            body_length = 100,
            vectors = 'askubuntu/vector/vectors_pruned.200.txt',
            questions_filename = 'askubuntu/text_tokenized.txt',
            dev_set = 'askubuntu/dev.txt'):

    vocabulary = Vocabulary(vectors)
    questions = QuestionBase(questions_filename, vocabulary, title_length, body_length)

    tester = TestFramework(dev_set, questions, title_length, body_length)

    # (samples) x (embedding_size)
    vectors = tester.sample(
        full_embedder
    )

    # Dot product matrix
    # (samples) x (samples)
    dot_products = torch.mm(vectors, vectors.transpose(0, 1))

    # Vector magnitude matrix
    magnitudes = (dot_products.diag() ** 0.5).view(-1, 1)
    denominators = torch.mm(magnitudes, magnitudes.transpose(0, 1))

    # Cosine similarity matrix
    cosine_similarities = torch.clamp(dot_products / denominators, min = -1, max = 1)
    angular_similarities = torch.acos(cosine_similarities) / math.pi

    # Run TSNE with cosine similarities as distances.
    tsne = TSNE(metric='precomputed')
    embedding = tsne.fit_transform(angular_similarities.data.cpu().numpy())

    # Load question file
    qdict = {}
    with open(questions_filename) as question_file:
        for line in question_file:
            qid, title, body = line.split('\t')
            qid = int(qid)
            qdict[qid] = title.decode('ascii', 'ignore')

    plt.figure(figsize=(20,20))
    plt.scatter(embedding[:, 0], embedding[:, 1])

    for i, question in enumerate(x['q'] for x in tester.test_set.entries):
        plt.annotate(qdict[question], (embedding[i, 0], embedding[i, 1]))

    plt.savefig(output)


sample(torch.load('models/best-gru-model/best.pkl'), 'models/tsne-plot.png')
