import numpy as np

from numpy import linalg as lin
from basics import cell_units as cu


class KohonenMap:

    def __init__(self, map_size, input_dim, weights=None, sigma0=None, tau2=1000, learning_rate0=0.1, method = 'som'):
        self.map_size = map_size
        self.weights = weights
        self.method = method
        self.input_dim = input_dim
        self.net = self._topology(map_size)
        self.tau2 = tau2
        self.step = 0
        self.X_ = None
        self.learning_rate0 = learning_rate0
        self.sigma0 = sigma0
        self.tau1 = 1000 / (np.log(sigma0))
        # self._create_lattice()
        # TODO assert for size(weights)

    def initialize(self):
        self.weights = np.random.uniform(0,1, (self.map_size, self.input_dim))
        self.step = 0

    def _topology(self, map_size):
        n = int(np.sqrt(map_size))
        m = map_size - n**2
        return np.ndarray((n + 1 if m else 0, n), dtype=cu.CellUnit)

    # def _create_lattice(self):
    #     for i in range(len(self.net)):
    #         for j in range(len(self.net[0])):
    #             if (i+1) * len(self.net[0]) + j < self.map_size:
    #                 self.net[i][j] = cu.CellUnit()
    #                 self.net[i][j].init(self.input_dim, 'x')
    #                 self.net[i][j].initialize_net()
    #
    # def _update_lattice(self):
    #     for i in range(len(self.net)):
    #         for j in range(len(self.net[0])):
    #             if (i+1) * len(self.net[0]) + j < self.map_size:
    #                 self.net[i][j].weights_ = self.weights[(i+1) * len(self.net[0]) + j]

    def learn(self):
        if self.method == 'som':
            self._learn_som()

    def put_data(self, X_):
        """
        :param X_: input learning vectors
        """
        self.X_ = X_.copy()

    def _learn_som(self):
        # 1. Initialization
        self.initialize()
        step = 1
        for x in self.X_:

            # 2. Competition
            winner_index = np.argmin([lin.norm(x - self.weights[i]) for i in range(self.map_size)])

            # 3. Cooperation
            for j in range(self.map_size):
                self.weights[j] += self._learning_rate(step) * self.h_j_i(step, self._wrap_coord(j), self._wrap_coord(
                    winner_index)) * (x - self.weights[j])
            step += 1

    def _wrap_coord(self, i):
        return np.array([i // len(self.net), i % len(self.net)])

    def d_j_i(self, j, i):
        return lin.norm(j - i)**2

    def _learning_rate(self, step):
        return self.learning_rate0 * np.exp(-step / self.tau2)

    def h_j_i(self, step, j, i):
        return np.exp(-self.d_j_i(j, i)**2 / (2 * self._sigma(step)**2))

    def _sigma(self, step):
        return self.sigma0 * np.exp(-step / self.tau1)

    def calc(self, x):
        return np.argmin([lin.norm(x - self.weights[i]) for i in range(self.map_size)])

if __name__ == '__main__':
    pass