import torch
torch.manual_seed(0)

from master_model import *
from data import *
from simple_models import *

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import itertools

import os
import json

version = 'third_versioned; batchnorm'

# Hyperparameters
def train(embedder,
            save_dir,
            title_length = 40,
            body_length = 100,
            batch_size = 100,
            margin = 0.1,
            epochs = 50,
            lr = 1e-4,
            cuda = True,
            body_embedder = None,
            merge_strategy = 'mean',
            negative_samples = 20,
            output_embedding_size = 0,
            alpha = 1e-4,
            vectors = 'glove/glove.i840B.300d.txt',
            source_filename = 'askubuntu/text_tokenized.txt',
            source_train_set = 'askubuntu/train_random.txt',
            target_filename = 'Android/corpus.tsv',
            dev_set = ('Android/dev.pos.txt', 'Android/dev.neg.txt'):

    vocabulary = Vocabulary(vectors, [source_filename, target_filename])
    source_questions = QuestionBase(source_filename, vocabulary, title_length, body_length)
    target_questions = QuestionBase(target_fiename, vocabulary, title_length, body_length)

    train_loader = DataLoader(
        TrainSet(source_train_set, source_questions),
        batch_size = batch_size,
        shuffle = True,
        drop_last = True
    )

    full_embedder = FullEmbedder(vocabulary, embedder, body_embedder, merge_strategy = merge_strategy, output_embedding_size = output_embedding_size)

    tester = AUROCTestFramework(dev_set, target_questions, title_length, body_length)
    master = CosineSimilarityMaster(full_embedder, title_length, body_length, margin)

    if cuda:
        master = master.cuda()


    # Initialize optimizers
    embed_optimizer = optim.Adam([param for param in master.parameters() if param.requires_grad], lr = lr)
    discrim_optimizer = optim.Adam([param for param in discriminator.parameters() if param.requires_grad])

    print('Parameter shapes:')
    print([param.size() for param in master.parameters() if param.requires_grad])
    print([param.size() for param in discriminator.parameters() if param.requires_grad])

    # Get total number of parameters
    product = lambda x: x[0] * product(x[1:]) if len(x) > 1 else x[0]
    number_of_parameters = sum(product(param.size()) for param in master.parameters() if param.requires_grad)

    # Make save directory
    if os.path.exists(save_dir):
        print('WARNING: saving to a directory that already has a model.')
    else:
        os.makedirs(save_dir)
    best_filename = os.path.join(save_dir, 'best.pkl')

    # Record model configuration and hyperparameters
    signature_filename = os.path.join(save_dir, 'signature.json')
    with open(signature_filename, 'w') as signature_file:
        json.dump({
            'title_length': title_length,
            'body_length': body_length,
            'merge_strategy': merge_strategy,
            'total_params': number_of_parameters,
            'batch_size': batch_size,
            'margin': margin,
            'epochs': epochs,
            'alpha': alpha,
            'negative_samples': negative_samples,
            'lr': lr,
            'cuda': cuda,
            'output_embedding_size': output_embedding_size,
            'body_embedder': (body_embedder.signature() if body_embedder is not None else None),
            'embedder': embedder.signature(),
            'vectors': vectors,
            'questions': questions_filename,
            'train_set': train_set,
            'dev_set': dev_set,
            'version': version
        }, signature_file)

    # Training loop
    best_loss = 0
    for epoch in range(epochs):
        final_loss = 0.0
        loss_denominator = 0.0

        master.train()

        for i, source_batch, discrim_batch in enumerate(tqdm(train_loader)):
            if cuda:
                batchify = lambda v: Variable(torch.stack(v).cuda())
            else:
                batchify = lambda v: Variable(torch.stack(v))

            _, random_indices = torch.rand(100).sort()

            indices = random_indices[:negative_samples]

            if cuda:
                indices = indices.cuda()

            # The batch will be an array of SimilarityEntry objects.
            # We need to merge these into appropriate LongTensor vectors.
            q, similar, random = (
                (batchify(source_batch['q_title']), batchify(source_batch['q_body'])),
                (batchify(source_batch['similar_title']), batchify(source_batch['similar_body'])),
                (batchify(source_batch['random_title'])[:, indices, :],
                    batchify(source_batch['random_body'])[:, indices, :])
            )

            # Run forward, backward.
            loss = master(q, similar, random).mean()
            regularization_factor = alpha * full_embedder.regularizer()
            final_loss += loss.data[0]
            loss += regularization_factor
            loss_denominator += 1

            loss.backward()

            # Step
            optimizer.step()

        master.eval()

        # Run test
        mean_average_precision, mean_reciprocal_rank, precision_at_n = tester.metrics(full_embedder)

        test_error = mean_reciprocal_rank

        print('Epoch %d: train hinge loss = %f, test MAP = %f, test MRR = %f \n, precision@1 = %f, precision@5 = %f' % (epoch, final_loss / loss_denominator, mean_average_precision, mean_reciprocal_rank, precision_at_n[0], precision_at_n[4]))

        save_filename = os.path.join(save_dir, 'epoch%d-loss%f-map%f.pkl' % (epoch, test_error, mean_average_precision))
        fig_filename = os.path.join(save_dir, 'epoch%d-loss%f-vectors.png' % (epoch, test_error))

        tester.visualize_embeddings(full_embedder, fig_filename)

        torch.save(full_embedder, save_filename)
        if test_error > best_loss:
            torch.save(full_embedder, best_filename)

        print('Epoch %d: train hinge loss %f, test MRR %0.1f' % (epoch, final_loss / loss_denominator, int(test_error * 1000) / 10.0))


#cnn_model = CNN()
gru_model = GRUAverage(hidden_size = 190, bidirectional = True, input_size = 302)

train(
    gru_model,
    'models/gru-unified-glove',
    batch_size = 50,
    lr = 1e-4,

    title_length = 40,
    negative_samples = 20,
    alpha = 0,

    vectors = 'glove/glove.840B.300d.txt',

    body_embedder = gru_model,
    body_length = 100,
    merge_strategy = 'mean',

    epochs = 50,
    margin = 0.2
)
