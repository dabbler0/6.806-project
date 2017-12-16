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
            samples_size = 100,
            questions_files = ['askubuntu/text_tokenized.txt'])

    vocabulary = Vocabulary(vectors)

    titles, bodies = [], []

    for question_file in questions_files:
        question_base = QuestionBase(questions_filename, vocabulary, title_length, body_length)
        title, body = question_base.get_random_batch(100)

        titles.append(title)
        bodies.append(body)

    title = torch.cat(titles, dim = 0)
    body = torch.cat(bodies, dim = 0)

    # (samples) x (embedding_size)
    vectors = full_embedder(title, body)

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

    colors = 'rgb'
    for i, _ in enumerate(questions_files):
        plt.scatter(embedding[(100 * i):(100 * (i + 1)), 0], embedding[(100 * i):(100 * (i + 1)), 1], c=colors[i])

    plt.savefig(output)


sample(torch.load('models/best-gru-model/best.pkl'), 'models/tsne-plot.png')
