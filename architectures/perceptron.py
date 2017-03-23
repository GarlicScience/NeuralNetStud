""" One layered rosenblatt perceptron.
"""


import numpy as np

from enum import Enum
from tools import activation_ as ac


_ACTIVATION = {'sgn': ac.sgn, 'logistic': ac.logistic}
_FIRST_DERIVATIVES = {'sgn': ac.dsgn, 'logistic': ac.dlogistic}


class PerceptronUnit:
    """
    Primitive of one perceptron.
    Attributes:

        input_size : int
            Size of input data vector

        function : {'tanh', 'sgn', 'logistic'}
            Activation function
    """

    def init(self, input_size, function):
        self.input_size = input_size
        self.function = function


def _add_logs(row, path='../examples/logs/lab1_log17'):
    """
    :param row: string to add
    Appends string to logfile
    """
    try:
        with open(path, 'a') as TextFile:
            print(row, file=TextFile)
    except FileNotFoundError:
        print("ERR. File is not found.")


def _clear_logs(path='../examples/logs/lab1_log17'):
    """
    Clears logfile
    """
    try:
        with open(path, 'w') as TextFile:
            TextFile.write(' ')
    except FileNotFoundError:
        print("ERR. File to clear is not found.")


class LearningResult(Enum):
    CONVERGENCE, DIVERGENCE = 0, 1


class RosenblattClassifier(PerceptronUnit):
    """
        Single-perceptron rosenblatt classifier.

        Parameters:

            input_size : int
                Size of input data vector

            function : {'tanh', 'sgn', 'logistic'}
                Activation function
                * default : 'sgn'

            learning_rate : float
                Weights changing rate
                * default : 0.001

            max_iter : int
                Maximum count of iterations in learning algorithm. The solver iterates until convergence
                (determined by 'tolerance') or this number of iterations
                * 0 value means having no limit for iterations
                * default : 0

            epsilon : float
                Required net() accuracy

        Attributes:

            weights_ : list of floats
                Weight coefficients for each input of sensor layer. Should be initialized
                * default : None

            bias_ : float
                Neuron bias value. Should be initialized
                * default : None

            loss_value_ : float
                Current value of loss function. Should be initialized
                * default : None

            n_iter_ : float
                Current vector number
                * default : None

            n_epoch_ : float
                Current epoch number
                * default : None

            X_ : 2d np_array of float
                Learning data set (list of input data vectors)
                * default : None

            d_ : np_array of float
                Expected net(x_)
                * default : None

            visualization_ : bool
                Weight saving on/off indicator
                * default : False

            log_ : bool
                Logging on/off indicator
                * default : False

            log_path : string
                Log-file path
                * default : '../examples/logs/lab1_log17'

            w_ : 2d list of weight history

    """

    def __init__(self, input_size, function='sgn', method='rosenblatt', learning_rate=0.001,
                 max_iter=0, epsilon=1e-3, alpha=1000, weights_=None, bias_=None):
        self.init(input_size, function)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.alpha = alpha
        self.X_ = None
        self.d_ = None
        self.weights_ = weights_
        self.bias_ = bias_
        self.method = method
        self.n_iter_ = None
        self.n_epoch_ = None
        self.loss_value_ = None
        self.visualization_ = 0
        self.log_ = False
        self.w_ = []
        self.log_path=None

    def initialize_net(self):
        """ Initializes:
             * weights as zero values list
             * biases as zero value
        """
        self.weights_ = np.zeros(self.input_size)
        self.bias_ = 0
        self.w_.append(np.append(self.weights_, [self.bias_]))
        self.n_iter_ = 1
        self.n_epoch_ = 0
        self.loss_value_ = None

    def put_data(self, X_, d_):
        """
        :param X_: input learning vectors
        :param d_: expected output vector
        """
        assert len(X_) == len(d_), 'Wrong data-set! ERR: Count of learning input vectors'
        self.X_ = X_.copy()
        self.d_ = d_.copy()

    def derivative(self, f, x, alpha=0.001):
        return _FIRST_DERIVATIVES[f](self._inducted_field(x), alpha)

    def net(self, x):
        """ Network trial method
        :param x : np.array
            Input vector
        :return: net(x) : 1 or 0
            Network answer
        """
        return _ACTIVATION[self.function](self._inducted_field(x), self.alpha)

    def _inducted_field(self, x):
        """ Inducted field counting method
        :param x: np.array(float)
        :return: inducted field by perceptron
        """
        x = np.array(x)
        assert type(x) == np.ndarray, 'ERR: Use numpy.array type for input vector instead of' + str(type(x))
        assert len(x) == len(self.weights_), 'ERR: Wrong dim(x): ' + str(len(x)) + ' must be ' + str(len(self.weights_))
        return self.weights_.dot(np.transpose(x)) + self.bias_

    def _init_visualisation(self, enable):
        self.visualization_ = True

    def set_log_(self, log_, log_path='../examples/logs/lab1_log17'):
        assert type(log_) == bool, 'ERR: Use bool for setting log_ value'
        self.log_ = log_
        self.log_path = log_path
        if log_:
            _clear_logs()

    def _pack(self):
        """ Turns learning set into a list of dicts = [{'x': X_[0], 'd': d_[0]}, {'x': X_[1], 'd': d_[1]}, ...]
        :return: pack : dict (see above)
        """
        assert np.ndim(self.X_) != len(self.d_), 'Wrong data-set! ERR: Count of learning input vectors'
        pack = []
        for i in range(len(self.d_)):
            pack.append({'x': np.array(self.X_[i]), 'd': self.d_[i]})
        return pack

    def learn(self):
        """Learning procedure. Executes learning methods up to set.
        :return LearningResult.CONVERGENCE or LearningResult.DIVERGENCE up to algorithm success"""
        assert self.function in _ACTIVATION.keys(), 'ERR: Wrong activation function'
        if self.method == 'rosenblatt':
            return self._learn_rosenblatt()

    def _learn_rosenblatt(self):
        """Sgn function type convergence learning"""
        assert self.X_ is not None, 'ERR: Define input vectors learning set'
        assert self.d_ is not None, 'ERR: Define output learning set'
        assert self.weights_ is not None, 'ERR: Weights are not initialized'
        assert self.bias_ is not None, 'ERR: Bias is not initialized'

        # 1. initialization
        self.n_iter_ = 0
        l_set = self._pack()

        # 1*. log initialization
        # TODO: create new log-files handling their count...
        log_file = None
        try:
            if self.log_:
                log_file = open(self.log_path, 'w')
        except FileNotFoundError:
            print("ERR. File is not found.")
            log_ = False

        while self.max_iter <= 0 or self.max_iter >= self.n_epoch_:
            for i in range(len(l_set)):
                self.n_epoch_ += 1
                self.n_iter_ %= len(l_set)
                self.n_iter_ += 1

                # 2. activation
                net = self.net(l_set[i]['x'])

                # 3. loss calculation
                loss_simple_ = l_set[i]['d'] - net

                # 3*. logging
                if self.log_:
                    print('Epoch ' + str(self.n_epoch_) + ' Vector_n = ' + str(self.n_iter_) + ' loss = ' +
                          str(loss_simple_) + ' weights = ' + str(self.weights_) + ' bias = ' + str(self.bias_),
                          file=log_file)

                # 4. checking end-loop condition
                if abs(loss_simple_) < self.epsilon:
                    continue

                # 5. weight adapting
                self.weights_ += self.learning_rate * loss_simple_ * \
                                 l_set[i]['x'] * self.derivative(self.function, l_set[i]['x'], self.alpha)
                self.bias_ += self.learning_rate * loss_simple_ * \
                              self.derivative(self.function, l_set[i]['x'], self.alpha)

                # 5. history saving
                if self.visualization_:
                    self.w_.append(np.append(self.weights_, [self.bias_]))

            # 6. calculation MSQ loss value
            self.loss_value_ = sum([(l_set[i]['d'] - self.net(l_set[i]['x'])) ** 2 for i in range(len(l_set))])

            # 6*. logging
            if self.log_:
                print('SQLoss = ', self.loss_value_, file=log_file)

            # 7. checking main end-loop condition
            if self.loss_value_ < self.epsilon:
                return LearningResult.CONVERGENCE

            # 8. shuffling learning set: randomized optimization
            np.random.shuffle(l_set)

        if self.log_:
            log_file.close()

        return LearningResult.DIVERGENCE

