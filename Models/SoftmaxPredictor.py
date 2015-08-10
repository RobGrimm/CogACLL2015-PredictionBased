import time
import numpy
import theano
import theano.tensor as T
from Layers.Softmax import Softmax
from Eval.functions import get_f1_and_classification_report


class SoftmaxPredictor(object):

    def __init__(self, data_set, n_next_words, learning_rate, input_size):
        """
        Wrapper class for a Softmax layer that predicts words from the right context and tweaks word embeddings on the
        basis of the prediction error. At each epoch, computes the accuracy of a 10-NN classifier trained on the
        embeddings. Stops training as soon as accuracy does not increase anymore.

        :type data_set:                 dict
        :param data_set:                data set with embeddings that are to be further modified during training.

        :type n_next_words:             int
        :param n_next_words:            number of words that are to predicted / output classes for the softmax model

        :type learning_rate:            float
        :param learning_rate:           learning rate for the softmax model

        :type input_size:               int
        :param input_size:              size of the word embeddings
        """
        self.data_set = data_set
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.n_next_words = n_next_words

        self.parameters = []                        # model parameters -- compute gradient with respect to them
        self.vocabulary = None
        self.embeddings_matrix = None
        self.softmax_layer = None
        self.cost_function = None

        # symbolic thenao variables needed for the theano train function
        self.y = T.ivector('y')                    # true class label
        self.y_idx = T.lscalar('y_idx')            # index pointing to true class label
        self.embedding_index = T.lscalar('index')  # index pointing to a row in the embeddings matrix

        self.training_error_over_epochs = []
        self.f1_over_epochs = []                   # micro f1 scores over epochs
        self.embeddings_over_epochs = []
        self.epochs = None                         # number of epochs the model was trained

        self._build_model()


    def _build_model(self):
        self.vocabulary = self.data_set['embeddings_dict'].keys()
        self.embeddings_matrix = self.data_set['embeddings_dict'].values()

        # cast embeddings matrix to theano type
        self.embeddings_matrix = theano.shared(numpy.asarray(self.embeddings_matrix, dtype=theano.config.floatX),
                                               borrow=True)
        # add embeddings matrix to list of parameters, to compute gradient later on
        self.parameters.append(self.embeddings_matrix)
        # a training example is a row in the embeddings matrix
        self.input_ = self.embeddings_matrix[self.embedding_index]
        # instantiate softmax model
        self.softmax_layer = Softmax(input=self.input_, n_in=self.input_size, n_out=self.n_next_words)
        # add softmax parameters (weight matrix) to self.parameters, to compute gradient later on
        self.parameters.extend(self.softmax_layer.params)
        self.cost_function = self.softmax_layer.negative_log_likelihood(self.y)


    def _get_train_function(self):
        # map each word from the vocabulary to a unique integer
        # (the index of the word's embedding in self.embeddings_matrix)
        l_to_idx = dict(zip(self.vocabulary, range(len(self.vocabulary))))
        train_set_idxs = [l_to_idx[w] for w in self.data_set['target_words']]

         # softmax classes to be predicted -- each class is a word from the right context
        train_set_y = [l_to_idx[w] for w in self.data_set['right_context_words']]

        # cast to theano type
        train_set_y = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)
        train_set_y = T.cast(train_set_y, 'int32')

        gr1 = T.grad(cost=self.cost_function, wrt=self.input_)
        gr2 = T.grad(cost=self.cost_function, wrt=self.softmax_layer.W)

        # this calculates gradients only for the row in the embeddings matrix that corresponds to the current input,
        # rather than calculating gradients for the entire matrix, len(matrix - 1) of which would be zero
        # see http://deeplearning.net/software/theano/tutorial/faq_tutorial.html, "How to update a subset of weights?"
        updates = [(self.embeddings_matrix, T.inc_subtensor(self.input_, - self.learning_rate * gr1)),
                   (self.softmax_layer.W, self.softmax_layer.W - self.learning_rate * gr2)]

        # train function takes an index pointing to a row in the embeddings matrix (self.embedding_index)
        # and an index pointing to an integer (true class label) in 'train_set_y' (self.y_idx)
        train_fn = theano.function(
            inputs=[self.embedding_index, self.y_idx],
            outputs=self.cost_function,
            givens={self.y: train_set_y[self.y_idx: self.y_idx + 1]},
            updates=updates,
            name='train'
        )

        return train_set_idxs, train_fn


    def train(self):

        print "There are %s training examples." % self.embeddings_matrix.get_value(borrow=True).shape[0]
        print

        # list of indices, each pointing to a row in the embeddings matrix
        train_set_idxs, train_fn = self._get_train_function()

        start_time = time.time()
        epoch = -1
        previous_accuracy = -numpy.inf
        while True:
            epoch += 1

            costs_over_batches = []
            for idx, y_idx in zip(train_set_idxs, range(len(train_set_idxs))):
                index = idx
                cost = train_fn(index, y_idx)
                costs_over_batches.append(cost)

            # store embeddings and evaluation metrics at each epoch
            training_loss = numpy.mean(costs_over_batches)
            embeddings = dict(zip(self.vocabulary, self.embeddings_matrix.get_value()))
            micro_f1, macro_f1, classification_report = get_f1_and_classification_report(embeddings, '10-NN')

            print 'epoch: %s -- loss: %s -- 10-NN accuracy: %s' % (epoch, training_loss, micro_f1)
            print classification_report
            print

            if micro_f1 <= previous_accuracy:
                print 'Accuracy did not increase relative to last epoch.'
                print 'I am aborting training and discarding results from the current epoch'
                break

            self.training_error_over_epochs.append(training_loss)
            self.f1_over_epochs.append(micro_f1)
            embeddings = dict(zip(self.vocabulary, self.embeddings_matrix.get_value()))
            self.embeddings_over_epochs.append(embeddings)
            self.epochs = epoch
            previous_accuracy = micro_f1

            print 'Time since beginning of training: %s' % ((time.time() - start_time) / 60)
            print

        print 'Training took: %s' % ((time.time() - start_time) / 60)