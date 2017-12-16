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

version = 'third_versioned; batchnorm'

# Hyperparameters
def train(embedder,
            save_dir,
            title_length = 40,
            body_length = 100,
            batch_size = 100,
            mixed_size = 50,
            test_batch_size = 100,
            margin = 0.1,
            epochs = 50,
            lr = 1e-4,
            cuda = True,
            body_embedder = None,
            merge_strategy = 'mean',
            negative_samples = 20,
            output_embedding_size = 0,
            alpha = 1e-4,
            discriminator = None,
            vectors = 'askubuntu/vector/vectors_pruned.200.txt',
            android_questions_filename = 'android/corpus.tsv',
            ubuntu_questions_filename = 'askubuntu/text_tokenized.txt',
            train_set = 'askubuntu/train_random.txt',
            dev_pos_txt = 'android/dev.pos.txt',
            dev_neg_txt = 'android/dev.neg.txt',
            test_neg_txt = 'android/test.neg.txt',
            test_pos_txt = 'android/test.pos.txt'):

    vocabulary = Vocabulary(vectors, [ubuntu_questions_filename, android_questions_filename])
    ubuntu_questions = QuestionBase(ubuntu_questions_filename, vocabulary, title_length, body_length)
    android_questions = QuestionBase(android_questions_filename, vocabulary, title_length, body_length)


    train_loader = DataLoader(
        TrainSet(train_set, ubuntu_questions),
        batch_size = batch_size,
        shuffle = True,
        drop_last = True
    )

    full_embedder = FullEmbedder(vocabulary, embedder, body_embedder, merge_strategy = merge_strategy, output_embedding_size = output_embedding_size)

    tester = AndroidTestFramework((dev_pos_txt, dev_neg_txt), android_questions, title_length, body_length, test_batch_size, num_examples = 100)
    master = CosineSimilarityMaster(full_embedder, title_length, body_length, margin)

    if cuda:
        master = master.cuda()
        discimrinator = discriminator.cuda()

    # Two optimizers, one which trains just the disciminator and one
    # which trains everything else
    optimizer = optim.Adam([param for param in master.parameters() if param.requires_grad], lr = lr)
    discrim_optimizer = optim.Adam([param for param in discriminator.parameters() if param.requires_grad], lr = -lr)

    # Get total number of parameters
    product = lambda x: x[0] * product(x[1:]) if len(x) > 1 else x[0]
    number_of_parameters = sum(product(param.size()) for param in master.parameters() if param.requires_grad)

    if number_of_parameters > 450000:
        print('WARNING: model with %d parameters is larger than the assignment permits.' % (number_of_parameters,))

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
            'questions': ubuntu_questions_filename,
            'train_set': train_set,
            'dev_set': dev_pos_txt,
            'version': version
        }, signature_file)

    best_loss = 0

    for epoch in range(epochs):
        final_loss = 0.0
        discrim_loss = 0.0
        loss_denominator = 0.0

        master.train()

        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            discrim_optimizer.zero_grad()

            if cuda:
                batchify = lambda v: Variable(torch.stack(v).cuda())
            else:
                batchify = lambda v: Variable(torch.stack(v))

            _, random_indices = torch.rand(100).cuda().sort()

            indices = random_indices[:negative_samples]

            # The batch will be an array of SimilarityEntry objects.
            # We need to merge these into appropriate LongTensor vectors.
            q, similar, random = (
                (batchify(batch['q_title']), batchify(batch['q_body'])),
                (batchify(batch['similar_title']), batchify(batch['similar_body'])),
                (batchify(batch['random_title'])[:, indices, :],
                    batchify(batch['random_body'])[:, indices, :])
            )

            # Run forward, backward.
            loss = master(q, similar, random).mean()

            # Now do the domain adaptation error.
            # Get some random vectors from each
            android_random_batch = android_questions.get_random_batch(mixed_size, cuda)
            ubuntu_random_batch = ubuntu_questions.get_random_batch(mixed_size, cuda)
            target = Variable(torch.LongTensor([0 for _ in range(mixed_size)] +
                [1 for _ in range(mixed_size)]))

            if cuda:
                target = target.cuda()

            # Embed and run through discriminator
            label_predictions = discriminator(torch.cat(
                [full_embedder(android_random_batch),
                full_embedder(ubuntu_random_batch)],
                dim = 0
            ))

            # Run the discriminator forward
            discrim_error = F.nll_loss(label_predictions, target)

            final_loss += loss.data[0]
            discrim_loss += discrim_error.data[0]

            # Subtract the discimrinator error to the label error
            loss -= alpha * discrim_error
            loss_denominator += 1

            loss.backward()

            # Step
            optimizer.step()
            discrim_optimizer.step()

        master.eval()

        # Run test
        AUC_metric = tester.metrics(full_embedder)
        print('Epoch %d: AUC score = %f' % (epoch, AUC_metric))

        save_filename = os.path.join(save_dir, 'epoch%d-loss%f.pkl' % (epoch, AUC_metric))
        fig_filename = os.path.join(save_dir, 'epoch%d-loss%f-vectors.png' % (epoch, AUC_metric))

        tester.visualize_embeddings(full_embedder, fig_filename)

        torch.save((discriminator, full_embedder), save_filename)
        if AUC_metric > best_loss:
            torch.save((discriminator, full_embedder), best_filename)

        print('Epoch %d: train hinge loss %f, discrim loss %f, test %0.1f' % (epoch, final_loss / loss_denominator, discrim_loss / loss_denominator, int(AUC_metric* 1000) / 10.0))

unified = GRUAverage(input_size = 302, hidden_size = 190)
train(
    embedder = unified,
    save_dir = 'models/gru-domain-adaptation-highest-alpha',
    batch_size = 100,
    test_batch_size = 10,
    lr = 3e-4,

    title_length = 40,
    negative_samples = 20,
    alpha = 1e-3,

    body_embedder = unified,
    body_length = 100,
    merge_strategy = 'mean',
    output_embedding_size = 400,
    vectors = 'glove/glove.840B.300d.txt',

    discriminator = TwoLayerDiscriminator(380, 380),

    epochs = 50,
    margin = 0.2
)
