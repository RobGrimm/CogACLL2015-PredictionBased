"""
Code is based on the Logistic Regression used in the Theano deep learning tutorial:
http://deeplearning.net/tutorial/code/logistic_sgd.py
"""
import numpy
import theano
import theano.tensor as T


class Softmax(object):

    def __init__(self, input, n_in, n_out):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """
        self.input = input

        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.params = [self.W]

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W))


    def get_predictions(self, input_):
        return T.nnet.softmax(T.dot(input_, self.W))


    def negative_log_likelihood(self, y):
        cost = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return cost