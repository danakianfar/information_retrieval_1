__author__ = 'agrotov'

import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
import query
import os
from scipy.sparse import dok_matrix
from scipy.special import expit
import matplotlib.pyplot as plt
import pandas as pd


NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

POINTWISE = 'pointwise'
PAIRWISE = 'pairwise'
LISTWISE = 'listwise'


# Cut-off level for NDCG metric
ndcg_k = 10

# Calculates the best NDCG@k for query with r relevant documents and binary relevance labels
def best_ndcg(r, k):
    if r == 0:
        raise ZeroDivisionError("No relevant documents for given query. NDCG can not be computed.")
    sum_limit = min(r, k)
    b_rank = 1 / np.log(1 + np.array(range(1, sum_limit + 1)))
    return b_rank, np.cumsum(b_rank, axis=0)

# Store the logarithmic discount and normalization factor lists for faster NDCG computation
disc_list, norm_list = best_ndcg(1000,1000)
disc_list = disc_list#.reshape((-1,1))

# Calculates the NDCG@k for a rank with binary relevance labels assuming a query with r relevant documents
def ndcg(rank, k, r = 1):
    return np.transpose(rank[:k]).dot(disc_list[:k]) / norm_list[r-1]

# Calculates the delta on the NDCG@1000 when documents at positions i and j are swapped
# id_i is the index in the labels list of the document in position i in the rank, sim. for id_j
def delta_ndcdg(i, j, id_i, id_j, labels):
    return (2**labels[id_i] - 2**labels[id_j]) * (disc_list[j] - disc_list[i]) * norm_list[int(np.sum(labels))]

# TODO: Implement the lambda loss function
def lambda_loss(output, lambdas):
    return lambdas.dot(output.reshape((-1,1)))

class LambdaRankHW:

    NUM_INSTANCES = count()

    def __init__(self, feature_count, measure_type = POINTWISE):
        self.feature_count = feature_count
        self.measure_type = measure_type
        self.output_layer = self.build_model(feature_count,1,BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)


    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs, val_queries, S):
        res = []
        try:
            now = time.time()
            for epoch in self.train(train_queries, val_queries, S):
                res.append(epoch)
                if epoch['number'] % 50 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}".format(epoch['train_loss']))
                    print("training mNDCG:\t\t{:.6f}".format(epoch['train_mndcg']))
                    print("validation mNDCG:\t\t{:.6f}\n".format(epoch['val_mndcg']))
                    now = time.time()
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass
        return res

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
        if self.measure_type == POINTWISE:
            # Point-wise loss function (squared error) - comment it out
            loss_train = lasagne.objectives.squared_error(output,y_batch)
        else:
            # Pairwise loss function - comment it in
            loss_train = lambda_loss(output,y_batch)

        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

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
    def lambda_function(self, labels, scores, S):
        lambda_vec = np.zeros((len(labels),1), dtype=np.float32)
        order = np.argsort(-scores)
        for ((w,l),_) in S.items():
            lambda_wl = - expit( scores[l] - scores[w])
            if self.measure_type == LISTWISE:
                lambda_wl *= delta_ndcdg(np.where(order == w)[0][0], np.where(order == l)[0][0], w, l, labels)
            lambda_vec[w,0] += lambda_wl
            lambda_vec[l,0] -= lambda_wl
        return lambda_vec


    def compute_lambdas_theano(self,query, labels, S):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)], S)
        return result

    def train_once(self, X_train, query, labels, S):

        # Alexandre L. correction
        resize_value = min(BATCH_SIZE, len(labels))
        X_train.resize((resize_value, self.feature_count), refcheck=False)

        # TODO: Comment out to obtain the lambdas

        if self.measure_type != POINTWISE:
            lambdas = self.compute_lambdas_theano(query,labels, S[query.get_qid()])
            lambdas.resize((resize_value, ))
            # TODO: Comment out (and comment in) to replace labels by lambdas
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        else:
            batch_train_loss = self.iter_funcs['train'](X_train, labels)

        return batch_train_loss


    def train(self, train_queries, val_queries, S):
        X_trains = train_queries.get_feature_vectors()

        queries = list(train_queries.values())

        for epoch in itertools.count(1):
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)

            # Calculates training loss
            batch_train_losses = []
            for index in range(len(queries)):
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()
                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels, S)
                batch_train_losses.append(batch_train_loss)
            avg_train_loss = np.mean(batch_train_losses)

            # Calculate NDCG on training set
            train_ndcgs = []
            queries = list(val_queries.values())
            for q in queries:
                q_scores = -self.score(q).flatten()
                sort_idx = np.argsort(q_scores)
                rank = labels[sort_idx]
                train_ndcgs.append(ndcg(rank, ndcg_k, int(np.sum(labels))))
            train_mndcg = np.mean(train_ndcgs)

            # Calculates mNDCG on validation set
            val_ndcgs = []
            queries = list(val_queries.values())
            for q in queries:
                labels = np.array(q.get_labels())
                q_scores = -self.score(q).flatten()
                sort_idx = np.argsort(q_scores)
                rank = labels[sort_idx]
                val_ndcgs.append(ndcg(rank, ndcg_k, int(np.sum(labels))))
            val_mndcg = np.mean(val_ndcgs)

            # Return statistics for current epoch
            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
                'val_mndcg': val_mndcg,
                'train_mndcg': train_mndcg
            }


def create_S_matrix(queries):
    S_dict = {}

    for query in queries.values():
        num_labels = query.get_document_count()
        S = dok_matrix((num_labels, num_labels), dtype=np.float32)
        labels = np.array(query.get_labels())
        ones_idx = np.where(labels == 1)[0]
        non_ones_idx = np.where(labels != 1)[0]
        for i in ones_idx:
            for j in non_ones_idx:
                try:
                    S[i, j] = 1
                except ValueError:
                    raise ValueError('%s, %s, %s, %s' % (query.get_qid(),i, j, num_labels))
        S_dict[query.get_qid()] = S

    return S_dict

def experiment(n_epochs, measure_type, num_features, num_folds):

    store_res = {}
    for fold in range(1,num_folds + 1):
        # Load queries from the corresponding fold
        print('\nLoading train queries')
        train_queries = query.load_queries(os.path.normpath('./HP2003/Fold%d/train.txt' % fold), num_features)

        print('\nLoading val queries')
        val_queries = query.load_queries(os.path.normpath('./HP2003/Fold%d/vali.txt' % fold), num_features)

        print('Creating the S Matrix')
        S = create_S_matrix({**train_queries, **val_queries})

        # Creates a new ranker
        ranker = LambdaRankHW(num_features, measure_type)

        # Stores the statistics for each epoch
        res = ranker.train_with_queries(train_queries, n_epochs, val_queries, S)

        # Saves the trained ranker
        res.append(ranker)

        # Stores the results for the current fold
        store_res[fold] = res

    return store_res
    #test_queries = query.load_queries(os.path.normpath('./HP2003/Fold%d/test.txt' % fold), num_features)

## Run
if __name__ == '__main__':
    n_epochs = 200
    measure_type = LISTWISE
    num_features = 64
    num_folds = 2

    res = experiment(n_epochs, measure_type, num_features, num_folds)
    ndcgs = [[rr['val_mndcg'] for rr in res[i][:-1]] for i in res]  # fold 1

    df = pd.DataFrame(ndcgs, index=['Fold%d' % d for d in range(1, num_folds + 1)]).T
    plt.plot(df)
    plt.legend(df.columns)
    plt.show()