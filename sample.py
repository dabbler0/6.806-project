'''
Sampling script that creates a TSNE plot of embeddings
for randomly drawn questions. Mostly for sanity-checking
and for fun to see how embeddings can reflect meaning.
'''

import torch
torch.manual_seed(0)

from master_model import *
from data import *
from architectures import *

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
            num_points = 100,
            vectors = 'askubuntu/vector/vectors_pruned.200.txt',
            questions_filename = 'askubuntu/text_tokenized.txt'):

    vocabulary = Vocabulary(vectors, [questions_filename], questions_filename)
    questions = QuestionBase(questions_filename, vocabulary, title_length, body_length)

    #tester = TestFramework(dev_set, questions, title_length, body_length)

    # (samples) x (embedding_size)
    qids, batch = questions.get_random_batch_and_qids(num_points)
    vectors = full_embedder(
        batch
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

    top_side = []
    bottom_side = []

    for i, question in enumerate(qids):
        plt.annotate(qdict[question].split(' '), (embedding[i, 0], embedding[i, 1]))

    plt.savefig(output)


if __name__ == '__main__':
    sample(torch.load('models/gru-summarizer-fixed/best.pkl')[0],
        'models/summarizer-tsne-plot.png',
        vectors = 'glove/glove.840B.300d.txt',
        questions_filename = 'Android/corpus.tsv',
        num_points = 500)
