import torch
torch.manual_seed(0)

from master_model import *
from data import *
from simple_models import *

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

# Hyperparameters
def train(embedder,
            truncate_length = 40,
            batch_size = 50,
            margin = 0.1,
            epochs = 100,
            lr = 0.0001,
            cuda = True,
            vectors = 'askubuntu/vector/vectors_pruned.200.txt',
            questions = 'askubuntu/text_tokenized.txt',
            train_set = 'askubuntu/train_random.txt',
            dev_set = 'askubuntu/dev.txt'):

    vocabulary = Vocabulary(vectors)
    questions = QuestionBase(questions, vocabulary, truncate_length)


    train_loader = DataLoader(
        TrainSet(train_set, questions),
        batch_size = batch_size,
        shuffle = True,
        drop_last = True
    )

    full_embedder = FullEmbedder(vocabulary, embedder)

    tester = TestFramework(dev_set, questions, truncate_length)
    master = CosineSimilarityMaster(full_embedder, truncate_length, margin)

    # if cuda:
    #     master = master.cuda()
    optimizer = optim.Adam([param for param in master.parameters() if param.requires_grad], lr = lr)

    for epoch in range(epochs):
        final_loss = 0.0
        loss_denominator = 0.0

        master.train()
        for batch in tqdm(train_loader):
            # The batch will be an array of SimilarityEntry objects.
            # We need to merge these into appropriate LongTensor vectors.

            q, similar, random = torch.stack(batch['q']), torch.stack(batch['similar']), torch.stack(batch['random'])

            # Transfer to GPU
            # if cuda:
            #     q, similar, random = q.cuda(), similar.cuda(), random.cuda()

            q, similar, random = Variable(q), Variable(similar), Variable(random)

            # Run forward, backward.
            loss = master(q, similar, random).mean()
            final_loss += loss.data[0]
            loss_denominator += 1

            loss.backward()
            # Step
            optimizer.step()

        master.eval()

        # Run test
        mean_average_precision, mean_reciprocal_rank, precision_at_n = tester.metrics(full_embedder)
        print('Epoch %d: train hinge loss = %f, test MAP = %f, test MRR = %f \n, precision@1 = %f, precision@5 = %f' % (epoch, final_loss / loss_denominator, mean_average_precision, mean_reciprocal_rank, precision_at_n[0], precision_at_n[4]))

train(
    #AverageEmbedding(),
    CNN(),
    #LSTMLast(),
    batch_size = 100,
    lr = 0.00001,
    truncate_length = 40,
    margin = 0.2
)
