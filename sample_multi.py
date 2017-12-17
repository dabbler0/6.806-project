'''
Sampling script that draws a TSNE plot of embeddings,
coloring questions from different corpora differently. For
domain adaptation sanity-checking.
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

def sample(full_embedder,
            output,
            title_length = 40,
            body_length = 100,
            vectors = 'glove/glove.840B.300d.txt',
            samples_size = 100,
            questions_files = ['askubuntu/text_tokenized.txt', 'android/corpus.tsv'],
            prune_corpora = ['android/corpus.tsv'],
            unprune_corpora = None):

    vocabulary = Vocabulary(vectors, prune_corpora, unprune_corpora)

    titles, bodies = [], []

    for question_file in questions_files:
        question_base = QuestionBase(question_file, vocabulary, title_length, body_length)
        title, body = question_base.get_random_batch(100)

        titles.append(title)
        bodies.append(body)

    title = torch.cat(titles, dim = 0)
    body = torch.cat(bodies, dim = 0)

    # (samples) x (embedding_size)
    vectors = full_embedder((title, body))

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

    plt.figure(figsize=(20,20))

    colors = 'rgb'
    for i, _ in enumerate(questions_files):
        plt.scatter(embedding[(100 * i):(100 * (i + 1)), 0], embedding[(100 * i):(100 * (i + 1)), 1], c=colors[i])

    plt.savefig(output)

if __name__ == '__main__':
    sample(torch.load('models/gru-dual-severe/best.pkl'), 'models/tsne-plot.png')
