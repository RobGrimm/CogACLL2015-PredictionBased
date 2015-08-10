import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from Layers.DenoisingAutoEncoder import DenoisingAutoEncoder
from Eval.functions import get_f1_and_classification_report


class BiModalAutoEncoder(object):

    def __init__(self, data_set, learning_rate, n_hidden, corruption_level, batch_size, context_feature_size,
                 phon_vectors, stress_vectors, phon_feature_size, stress_feature_size):
        """
        Wrapper class for an Auto Encoder that projects bimodal input (contextual and phonological) onto a hidden layer.
        At each epoch, computes the accuracy of a 10-NN classifier trained on the vectors of hidden unit activations.
        Stops training if accuracy has not surpassed the previous best score for 50 epochs.

        :type data_set:                 dict
        :param data_set:                ata set. contains vectors of co-occurrence counts for the left
                                        contexts of target words

        :type learning_rate:            float
        :param learning_rate:           learning rate for the auto encoder

        :type n_hidden:                 int
        :param n_hidden:                number of hidden units

        :type corruption_level:         float
        :param corruption_level:        probability with which each visible units is set to zero before processing a
                                        given training example

        :type batch_size:               int
        :param model_name:              training batch size

        :type context_feature_size:     int
        :param context_feature_size:    size of the context feature vectors (equal to vocabulary size)

        :type phon_vectors:             dictionary
        :param phon_vectors:            keys: word-strings, values: phonological feature vectors

        :type stress_vectors:           dictionary
        :param stress_vectors:          keys: word-strings, values: lexical stress feature vectors

        :type phon_feature_size:        int
        :param phon_feature_size:       size of the phonological feature vectors

        :type stress_feature_size:      int
        :param stress_feature_size:     size of the lexical stress feature vectors
        """
        self.data_set = data_set
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.corruption_level = corruption_level
        self.batch_size = batch_size

        self.phon_vectors = phon_vectors
        self.primary_stress_vectors = stress_vectors
        self.phon_component_size = phon_feature_size
        self.primary_stress_component_size = stress_feature_size
        self.context_component_size = context_feature_size

        # a training example is a concatenation of one type of feature vector each,
        # so the total input size is the sum of feature vector sizes
        self.input_size = self.context_component_size + self.phon_component_size + self.primary_stress_component_size

        self.vocabulary = None
        self.embeddings_matrix = None
        self.auto_encoder = None

        self.embeddings_over_epochs = []        # keeps track of the embeddings at each epoch
        self.training_error_over_epochs = []    # training error over epochs
        self.f1_over_epochs = []                # micro f1 scores over epochs
        self.embeddings_over_epochs = []        # keeps track of hidden embeddings over epochs
        self.epochs = None                      # number of epochs the model was trained

        self._set_visible_vectors()


    def train(self):

        n_train_batches = self.embeddings_matrix.get_value(borrow=True).shape[0] / self.batch_size
        print 'Nr of train batches: %s' % n_train_batches
        print

        train_function = self._get_train_function()
        start_time = time.time()

        best_accuracy = -numpy.inf
        n_epochs_without_improvement = 0
        epoch = 0

        while True:
            costs_over_batches = []
            for batch_index in xrange(n_train_batches):
                costs_over_batches.append(train_function(batch_index))

            # store evaluation metrics at every epoch
            avg_cost = numpy.mean(costs_over_batches)
            hidden_vectors = self.get_hidden_vectors()
            self.embeddings_over_epochs.append(hidden_vectors)
            micro_f1, macro_f1, classification_report = get_f1_and_classification_report(hidden_vectors, '10-NN')
            self.training_error_over_epochs.append(avg_cost)
            self.f1_over_epochs.append(micro_f1)

            print 'epoch: %s -- loss: %s' % (epoch, avg_cost)
            print 'Embeddings -- 10-NN accuracy: %s' % micro_f1
            print 'Time since beginning of training: %s' % ((time.time() - start_time) / 60)
            print

            if micro_f1 > best_accuracy:
                best_accuracy = micro_f1
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

            if n_epochs_without_improvement == 10:
                print 'Accuracy did not exceed best previous score for 50 epochs.'
                print 'I am aborting training and discarding results from the current epoch'
                print 'The best accuracy score is: %s (epoch %s)' % (best_accuracy, epoch - 10)
                print

                self.epochs = epoch - 10

                # discard results from last 10 epochs
                self.training_error_over_epochs = self.training_error_over_epochs[:-10]
                self.f1_over_epochs = self.f1_over_epochs[:-10]
                self.embeddings_over_epochs = self.embeddings_over_epochs[:-10]
                break

            epoch += 1

        print 'Training took: %s' % ((time.time() - start_time) / 60)


    def get_hidden_vectors(self):
        """ for each embedding, get a vector of hidden unit activation values from the auto encoder """
        assert self.auto_encoder is not None
        hidden_vectors = self.auto_encoder.get_hidden(self.embeddings_matrix).eval()
        return dict(zip(self.vocabulary, hidden_vectors))


    def _get_train_function(self):

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        x = T.matrix('x')

        self.auto_encoder = DenoisingAutoEncoder(numpy_rng=rng,
                                                 theano_rng=theano_rng,
                                                 input_=x,
                                                 n_visible=self.input_size,
                                                 n_hidden=self.n_hidden)

        cost, updates = self.auto_encoder.get_cost_updates(self.corruption_level, self.learning_rate)

        index = T.lscalar()
        train_fn = theano.function([index], cost, updates=updates,
                                   givens={x: self.embeddings_matrix[index *
                                                                     self.batch_size: (index + 1) * self.batch_size]})
        return train_fn


    def _set_visible_vectors(self):
        """ load data set, add phonological / lexical stress feature vectors, cast embeddings matrix to theano type """
        self.embeddings_matrix = self.data_set['embeddings_dict'].values()
        # word-labels for each row in embeddings matrix
        self.vocabulary = self.data_set['embeddings_dict'].keys()
        # concatenate embeddings with phonological / lexical stress feature vectors, if present
        self._add_phonological_components()
        # cast embeddings matrix to theano type
        self._embeddings_matrix_to_theano()


    def _add_phonological_components(self):
        """" if present, add phonological and stress feature vectors to embeddings """
        assert len(self.embeddings_matrix) == len(self.vocabulary)
        concatenated = [[] for w in self.vocabulary]

        if self.primary_stress_vectors is not None:
            concatenated = [v + self.primary_stress_vectors[w] for v, w in zip(concatenated, self.vocabulary)]

        if self.phon_vectors is not None:
            concatenated = [v + list(self.phon_vectors[w]) for v, w in zip(concatenated, self.vocabulary)]

        self.embeddings_matrix = [numpy.concatenate([v1, v2]) for v1, v2 in zip(self.embeddings_matrix, concatenated)]


    def _embeddings_matrix_to_theano(self):
        """ cast embeddings matrix to theano type """
        self.embeddings_matrix = theano.shared(numpy.asarray(self.embeddings_matrix, dtype=theano.config.floatX),
                                               borrow=True)