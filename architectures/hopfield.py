import numpy as np
from tools import activation_ as ac


class HopfieldNet:

    def __init__(self, neuron_count, weights=None):
        self.neuron_count_ = neuron_count
        self.weights_ = weights
        # TODO: assert for W matrix
        self.fundamentals_ = None
        # TODO: assert for +-1 vectors elements equality

    def learn(self):
        assert self.fundamentals_ is not None, 'ER01. Put fundamentals!'
        self.weights_ = np.ndarray((len(self.fundamentals_), self.neuron_count_))
        for i in range(self.neuron_count_):
            for j in range(self.neuron_count_):
                if i != j:
                    self.weights_[j][i] =  sum([self.fundamentals_[mu][j] * self.fundamentals_[mu][i]
                                            for mu in range(len(self.fundamentals_))]) / self.neuron_count_
                else: self.weights_[j][i] = 0

    def calc(self, ksi):
        ksi_new = [ac.ACTIVATION['sign'](sum([self.weights_[j][i] * ksi[j] for i in range(self.neuron_count_)])) for j in range(len(self.fundamentals_))]
        return ksi_new if ksi == ksi_new else self.calc(ksi_new)

    def put_data(self, fundamentals_):
        """
        :param X_: input learning vectors
        """
        self.fundamentals_ = fundamentals_.copy()