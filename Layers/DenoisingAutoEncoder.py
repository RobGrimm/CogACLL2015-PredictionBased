"""
Based on the denoising Auto Encoder used in the Theano deep learning tutorial:
http://deeplearning.net/tutorial/code/dA.py
"""
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class DenoisingAutoEncoder(object):

    def __init__(self,
            numpy_rng,
            n_visible,
            n_hidden,
            theano_rng=None,
            input_=None,
            W=None,
            bhid=None,
            bvis=None,
            activation_fn='truncated_lin_rect'):
        """
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input_: theano.tensor.TensorType
        :param input_: a symbolic description of the input or None for
                      standalone dA

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type activation_fn: str
        :param activation_fn: the activation function to be used at the hidden layer.
                              Must be one of: 'lin_rect', 'sigmoid', 'truncated_lin_rect'
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if activation_fn not in ['lin_rect', 'sigmoid', 'truncated_lin_rect']:
            raise Exception('Unknown activation function: %s' % activation_fn)

        self.activation_fn = activation_fn

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                                                        high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                                                        size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX), borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='b', borrow=True)

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng

        if input_ is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input_

        self.params = [self.W, self.b, self.b_prime]
        self.output = self.get_hidden(self.x)


    def _get_corrupted_input(self, input_, corruption_level):
        return self.theano_rng.binomial(size=input_.shape, n=1, p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input_


    def get_hidden(self, input_):
        """ Computes the values of the hidden units """
        hidden = T.maximum(0, T.minimum(1, T.dot(input_, self.W) + self.b))
        return hidden


    def get_hidden_non_theano(self, input):
        weight_matrix = self.W.get_value(borrow=True)
        bias = self.b.get_value(borrow=True)
        pre_activation = numpy.dot(input, weight_matrix) + bias
        hidden = numpy.array([max(0, min(1, i)) for i in pre_activation])
        return hidden


    def _get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the hidden units """
        a = T.dot(hidden, self.W_prime) + self.b_prime
        return T.nnet.sigmoid(a)


    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self._get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden(tilde_x)
        z = self._get_reconstructed_input(y)

        # euclidean distance
        L = T.sqrt(T.sum(T.sqr(self.x - z), axis=1))
        # squared error
        L = L * L
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return cost, updates