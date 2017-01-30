__author__ = 'agrotov'

import itertools
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
from itertools import count
import query

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

POINTWISE = 'pointwise'
PAIRWISE = 'pairwise'
LISTWISE = 'listwise'

BEST_NDCG_10 = 140.850339481

## Utility methods

def dcg(ranked_labels, k):
    gains = [(2**ranked_labels[i] -1 ) / (np.log2(2 + i)) for i in range(k)]
    return sum(gains)

def ndcg(ranking, k, best_ndcg_k):
  return dcg(ranking, k)/best_ndcg_k



# # TODO: Implement the lambda loss function
def lambda_loss(output, lambdas):
    raise "Unimplemented"


class LambdaRankHW:

    NUM_INSTANCES = count()

    def __init__(self, feature_count, type):
        self.measure_type = type
        self.feature_count = feature_count
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        try:
            for epoch in self.train(train_queries):
                if epoch['number'] % 10 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print("input_dim",input_dim, "output_dim",output_dim)
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )


        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")

        # TODO: Change loss function
        # Point-wise loss function (squared error) - comment it out
        loss_train = lasagne.objectives.squared_error(output,y_batch)
        # Pairwise loss function - comment it in
        # loss_train = lambda_loss(output,y_batch)

        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)


        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch],output_row_det,
        )

        # (2) Training function, updates the parameters, outpust loss
        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,
            # givens={
            #     X_batch: dataset['X_train'][batch_slice],
            #     # y_batch: dataset['y_valid'][batch_slice],
            # },
        )

        print("finished create_iter_functions")
        return dict(
            train=train_func,
            out=score_func,
        )

    # TODO: Implement the aggregate (i.e. per document) lambda function
    def lambda_function(self,labels, scores):
        pass


    def compute_lambdas_theano(self,query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        return result

    def train_once(self, X_train, query, labels):

        # TODO: Comment out to obtain the lambdas
        # lambdas = self.compute_lambdas_theano(query,labels)
        # lambdas.resize((BATCH_SIZE, ))

        #X_train.resize((BATCH_SIZE, self.feature_count),refcheck=False)
        # Alexandre L. correction
        resize_value = BATCH_SIZE
        if self.measure_type == POINTWISE:
            resize_value = min(resize_value, len(labels))
        X_train.resize((resize_value, self.feature_count), refcheck=False)

        # TODO: Comment out (and comment in) to replace labels by lambdas
        #batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        batch_train_loss = self.iter_funcs['train'](X_train, labels)
        return batch_train_loss


    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()

        queries = list(train_queries.values())

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in range(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()

                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                batch_train_losses.append(batch_train_loss)


            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }


from operator import itemgetter
import os
import time

def experiment(experiment_type):
    print('- Running', experiment_type)
    n_features = 64

    # Implements 5-Folds validation
    kfold_ndcg = []
    for i in range(1, 6):
        ranker = LambdaRankHW(n_features, experiment_type)#, type=experiment_type)
        n_epochs = 5

        def query_ndcg(q):
            scores = ranker.score(q).flatten()
            labels = q.get_labels()
            return ndcg(list(zip(*sorted(list(zip(labels, scores)), key=itemgetter(1), reverse=True)))[0])

        for j in range(1, 6):
            if i == j:
                continue
            start_time = time.time()
            queries = query.load_queries(os.path.normpath('./HP2003/Fold%d/train.txt' % j), n_features)
            ranker.train_with_queries(queries, n_epochs)
            print('Elapsed time', time.time() - start_time)

        queries = query.load_queries(os.path.normpath('./HP2003/Fold%d/train.txt' % i), n_features)
        kfold_ndcg.append(np.mean([query_ndcg(q) for q in queries]))
        print("- mNDCG: %.3f" % kfold_ndcg[-1])

    print("- Average mNDCG: %.3f" % np.average(kfold_ndcg))


import sys

if __name__ == '__main__':
    # sys.argv[1] can be 'pointwise', 'pairwise' or 'listwise'
    experiment(POINTWISE)