import torch
torch.manual_seed(0)

from master_model import *
from data import *
from architectures import *

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

import os
import json

version = 'third_versioned; batchnorm'

# Hyperparameters
def train(
            save_dir,
            title_length = 40,
            body_length = 100,
            batch_size = 100,
            test_batch_size = 100,
            epochs = 50,
            lr = 1e-4,
            cuda = True,
            alpha = 1e-4,
            vectors = 'askubuntu/vector/vectors_pruned.200.txt',
            android_questions_filename = 'android/corpus.tsv',
            questions_filename = 'android/corpus.tsv',
            dev_pos_txt = 'android/dev.pos.txt',
            dev_neg_txt = 'android/dev.neg.txt'):

    vocabulary = Vocabulary(vectors, [android_questions_filename], android_questions_filename)
    questions = QuestionBase(questions_filename, vocabulary, title_length, body_length)

    encoder = GRUAverage(input_size = 302, hidden_size = 190)

    full_embedder = BodyOnlyEmbedder(vocabulary, encoder)

    decoder = GRUDecoder(full_embedder.word_embedding, input_size = 302 + 380, hidden_size = 190, output_size = len(vocabulary.vocabulary))

    tester = AndroidTestFramework((dev_pos_txt, dev_neg_txt), questions, title_length, body_length, test_batch_size, num_examples = 100)

    if cuda:
        full_embedder = full_embedder.cuda()
        decoder = decoder.cuda()

    optimizer = optim.Adam(
        [param for param in full_embedder.parameters() if param.requires_grad] +
        [param for param in decoder.parameters() if param.requires_grad],
        lr = lr
    )

    # Get total number of parameters
    product = lambda x: x[0] * product(x[1:]) if len(x) > 1 else x[0]
    # number_of_parameters = sum(product(param.size()) for param in master.parameters() if param.requires_grad)

    # if number_of_parameters > 450000:
    #     print('WARNING: model with %d parameters is larger than the assignment permits.' % (number_of_parameters,))

    if os.path.exists(save_dir):
        print('WARNING: saving to a directory that already has a model.')
    else:
        os.makedirs(save_dir)
    best_filename = os.path.join(save_dir, 'best.pkl')

    signature_filename = os.path.join(save_dir, 'signature.json')

    with open(signature_filename, 'w') as signature_file:
        json.dump({
            'title_length': title_length,
            'body_length': body_length,
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'cuda': cuda,
            'encoder': encoder.signature(),
            'decoder': decoder.signature(),
            'vectors': vectors,
            'dev_set': dev_pos_txt,
            'version': version
        }, signature_file)

    best_loss = 0

    for epoch in range(epochs):
        final_loss = 0.0
        loss_denominator = 0.0

        full_embedder.train()
        decoder.train()

        # Some arbitrary batch sizes
        for _ in tqdm(range(100), total=100, desc='batches'):
            optimizer.zero_grad()

            titles, bodies = questions.get_random_batch(batch_size)

            loss = decoder(full_embedder((None, bodies)), titles)

            final_loss += loss.data[0]
            loss_denominator += 1

            loss.backward()

            optimizer.step()

        full_embedder.eval()
        decoder.eval()

        # Run test
        AUC_metric = tester.metrics(full_embedder)

        save_filename = os.path.join(save_dir, 'epoch%d-loss%f.pkl' % (epoch, AUC_metric))
        fig_filename = os.path.join(save_dir, 'epoch%d-loss%f-vectors.png' % (epoch, AUC_metric))

        # Visualize the vector embeddings, as a sanity check
        # so that we can see if the vectors are all the same,
        # or some other pathological case like that.
        tester.visualize_embeddings(full_embedder, fig_filename)

        torch.save((full_embedder, decoder), save_filename)
        if AUC_metric > best_loss:
            torch.save((full_embedder, decoder), best_filename)

        print('Epoch %d: train loss %f, dev AUC %0.1f' %
            (epoch, final_loss / loss_denominator, int(AUC_metric* 1000) / 10.0))

if __name__ == '__main__':
    # Nothing really worked here
    train(
        save_dir = 'models/gru-summarizer-interlinear',
        batch_size = 100,
        test_batch_size = 10,
        lr = 3e-4,

        title_length = 40,

        body_length = 100,
        vectors = 'glove/glove.840B.300d.txt',

        epochs = 50
    )
