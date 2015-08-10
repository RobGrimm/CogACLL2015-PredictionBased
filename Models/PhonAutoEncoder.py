import random
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from Layers.DenoisingAutoEncoder import DenoisingAutoEncoder
from Phonology.get_phonology_vectors import get_phoneme_vectors


class PhonAutoEncoder(object):

    def __init__(self, vocabulary, learning_rate, n_hidden, corruption_level, epochs, batch_size, input_size=114):
        """
        Wrapper class for an Auto Encoder that projects phonological feature vectors onto a hidden layer.

        :type vocabulary:           array
        :param vocabulary:          array of strings -- words whose phonological feature vectors you want to train on

        :type learning_rate:        float
        :param learning_rate:       learning rate for the auto encoder

        :type n_hidden:             int
        :param n_hidden:            number of hidden units

        :type corruption_level:     float
        :param corruption_level:    probability with which each visible units is set to zero

        :type epochs:               int
        :param epochs:              number of training epochs

        :type batch_size:           int
        :param batch_size:          training batch size
        """
        self.vocabulary = vocabulary
        self.input_size = input_size    # each phonological feature vector is 114-dimensional (default value)
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.corruption_level = corruption_level
        self.epochs = epochs
        self.batch_size = batch_size
        self.auto_encoder = None
        self.word_phon_mappings = None  # will be a dictionary mapping each word in 'self.words' to its feature vector

        self._set_visible_vectors()


    def _get_auto_encoder_and_train_function(self, train_set_vs):

        index = T.lscalar()
        x = T.matrix('x')
        rng = numpy.random.RandomState(999)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.auto_encoder = DenoisingAutoEncoder(numpy_rng=rng,
                                       theano_rng=theano_rng,
                                       input_=x,
                                       n_visible=self.input_size,
                                       n_hidden=self.n_hidden)

        cost, updates = self.auto_encoder.get_cost_updates(self.corruption_level, self.learning_rate)

        train_fn = theano.function([index], cost, updates=updates,
                                   givens={x: train_set_vs[index * self.batch_size: (index + 1) * self.batch_size]})
        return train_fn


    def train(self):

        visible_vectors, labels = self.visible_vectors

        print 'Number of training examples: %s' % len(labels)
        print

        n_train_batches = visible_vectors.get_value(borrow=True).shape[0] / self.batch_size
        print 'Number of train batches: %s' % n_train_batches

        train_da = self._get_auto_encoder_and_train_function(visible_vectors)

        mean_cost = []
        for epoch in xrange(self.epochs):

            for batch_index in xrange(n_train_batches):
                mean_cost += [train_da(batch_index)]

            avg_cross_entropy = numpy.mean(mean_cost)

            print 'epoch: %s -- loss: %s' % (epoch, avg_cross_entropy)
            print


    def get_hidden_vectors(self):
        # for each feature vector, get a vector
        # of hidden unit activation values from the auto encoder
        assert self.auto_encoder is not None
        hidden_vectors = dict()
        for w, v in self.word_phon_mappings.items():
            hidden_vectors[w] = self.auto_encoder.get_hidden(v).eval()
        return hidden_vectors


    def _set_visible_vectors(self):
        # get the phonological feature vectors for each of the words in self.words
        self.word_phon_mappings = get_phoneme_vectors(self.vocabulary, left=False)
        mapping = self.word_phon_mappings.items()
        # randomize order of training examples
        random.seed(123)
        random.shuffle(mapping)
        visible_vectors = [v for w, v in mapping]
        # cast feature vectors to theano type
        visible_vectors = theano.shared(numpy.asarray(visible_vectors, dtype=theano.config.floatX), borrow=True)
        labels = [w for w, v in mapping]
        self.visible_vectors = [visible_vectors, labels]