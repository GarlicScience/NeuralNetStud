from tools import activation_ as ac
import numpy as np


class CellUnit:
    """
    Primitive of one perceptron.
    Attributes:

        function : {'tanh', 'sgn', 'logistic'}
            Activation function
            * default : 'sgn'

        weights_ : list of floats
            Weight coefficients for each input of sensor layer. Should be initialized
            * default : None

        bias_ : float
            Neuron bias value. Should be initialized
            * default : None

        input_size : int
            Size of input data vector

        function : {'tanh', 'sgn', 'logistic'}
            Activation function
    """
    input_size = None
    function = None
    weights_ = None
    bias_ = None
    alpha = None

    def init(self, input_size, function, weights_=None, bias_=None, alpha=1):
        self.input_size = input_size
        self.function = function
        self.weights_ = weights_
        self.bias_ = bias_
        self.alpha = alpha

    def calc(self, x):
        """ Unit's calculation method
        :param x : np.array
            Input vector
        :return: net(x) : 1 or 0
            Network answer
        """
        return ac.ACTIVATION[self.function](self._inducted_field(x), self.alpha)

    def _inducted_field(self, x):
        """ Inducted field counting method
        :param x: np.array(float)
        :return: inducted field by perceptron
        """
        x = np.array(x)
        assert type(x) == np.ndarray, 'ERR: Use numpy.array type for input vector instead of' + str(type(x))
        assert len(x) == len(self.weights_), 'ERR: Wrong dim(x): ' + str(len(x)) + ' must be ' + str(len(self.weights_))
        return self.weights_.dot(np.transpose(x)) + self.bias_
