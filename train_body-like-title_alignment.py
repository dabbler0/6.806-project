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

version = 'submitted'

# Hyperparameters
def train(
            save_dir,
            schedule = lambda p: 2 * p,
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

    title_encoder = GRUAverage(input_size = 302, hidden_size = 190)
    body_encoder = GRUAverage(input_size = 302, hidden_size = 190)

    full_embedder = FullEmbedder(vocabulary, title_encoder, body_encoder, merge_strategy = 'concatenate')

    tester = AndroidTestFramework((dev_pos_txt, dev_neg_txt), questions, title_length, body_length, test_batch_size, num_examples = 100)

    if cuda:
        full_embedder = full_embedder.cuda()

    optimizer = optim.Adam(
        [param for param in full_embedder.parameters() if param.requires_grad],
        lr = lr
    )

    # Get total number of parameters
    product = lambda x: x[0] * product(x[1:]) if len(x) > 1 else x[0]

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
            'title': title_encoder.signature(),
            'body': body_encoder.signature(),
            'vectors': vectors,
            'dev_set': dev_pos_txt,
            'version': version
        }, signature_file)

    best_loss = 0

    for epoch in range(epochs):
        final_loss = 0.0
        final_reg = 0.0
        loss_denominator = 0.0

        full_embedder.train()
        # Some arbitrary batch sizes
        for _ in tqdm(range(100), total=100, desc='batches'):
            optimizer.zero_grad()

            # Embed these things
            # (batch_size) x (embedding_size)
            embeddings = full_embedder(questions.get_random_batch(batch_size))

            # Get the individual title and body embeddings
            title_embeddings = embeddings[:, :380]
            body_embeddings = embeddings[:, 380:]

            # These are each (batch_size) x (embedding_size)
            # Take all these dot-products
            title_dots = torch.mm(title_embeddings, title_embeddings.transpose(0, 1))
            body_dots = torch.mm(body_embeddings, body_embeddings.transpose(0, 1))

            # Take cosine similarities
            # Denominator is (batch_size) x 1 X 1 x (batch_size)
            # For outer product
            title_mags = title_dots.diag().unsqueeze(1) ** 0.5
            title_denoms = torch.mm(title_mags, title_mags.transpose(0, 1))
            body_mags = body_dots.diag().unsqueeze(1) ** 0.5
            body_denoms = torch.mm(body_mags, body_mags.transpose(0, 1))

            title_sims = title_dots / title_mags
            body_sims = body_dots / body_mags

            # We want to minimize similarities
            # But make the title and body have similar similarities
            # to each other.
            loss = title_sims.mean() + body_sims.mean()
            reg = ((title_sims - body_sims) ** 2).mean()

            final_loss += loss.data[0]
            final_reg += reg.data[0]

            # Move lambda according to schedule
            l = schedule(float(epoch) / epochs)

            loss += l * reg

            loss.backward()

            loss_denominator += 1

            optimizer.step()

        # Free memory
        del title_embeddings, body_embeddings, title_dots, body_dots, title_mags, title_denoms, body_mags, body_denoms, title_sims, body_sims, loss

        full_embedder.eval()

        # Run test
        AUC_metric = tester.metrics(full_embedder)

        save_filename = os.path.join(save_dir, 'epoch%d-loss%f.pkl' % (epoch, AUC_metric))
        fig_filename = os.path.join(save_dir, 'epoch%d-loss%f-vectors.png' % (epoch, AUC_metric))

        # Visualize the vector embeddings, as a sanity check
        # so that we can see if the vectors are all the same,
        # or some other pathological case like that.
        tester.visualize_embeddings(full_embedder, fig_filename)

        torch.save(full_embedder, save_filename)
        if AUC_metric > best_loss:
            torch.save(full_embedder, best_filename)

        print('Epoch %d: train dist %f, train reg loss %f, test AUC %0.1f' % (epoch, final_loss / loss_denominator, final_reg / loss_denominator, int(AUC_metric* 1000) / 10.0))

if __name__ == '__main__':
    # The best model had these hyperparameters.
    train(
        save_dir = 'models/gru-dual-severe',
        batch_size = 100,
        test_batch_size = 10,
        lr = 3e-4,

        schedule = lambda x: 10 * x + 1,

        title_length = 40,

        body_length = 100,
        vectors = 'glove/glove.840B.300d.txt',

        epochs = 20
    )
