import numpy as np

from scipy import linalg as lin
from tools import data_ as dt
from basics import cell_units as cu
from tools import activation_ as ac
import pylab as plt
"""
    RBF network solution
"""

RBF_FUNCTIONS ={'rbf_gauss', 'rbf_green'}


class RBFNetwork:

    def __init__(self, input_size, hidden_layer_size, learning_method='sorg_green', function='rbf_green', learning_rate_weight=0.1,
                 learning_rate_center_position=0.1, learning_rate_sigma=0.1,
                 max_iter=0, epsilon=1e-3, weights_=None, bias_=None, sigma = None):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate_weight = learning_rate_weight
        self.learning_rate_center_position = learning_rate_center_position
        self.learning_rate_sigma = learning_rate_sigma
        self.learning_method = learning_method
        self.sigma = sigma
        self.rbf_func = function
        self.weights_ = weights_
        self.bias_ = bias_
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.X_ = None
        self.T_ = None
        self.d_ = None
        self.n_iter_ = None
        self.n_epoch_ = None
        self.loss_value_ = None
        self.visualization_ = 0
        self.log_ = False
        self.w_ = []
        self.log_path = None
        self.hidden_layer = []
        self.output_cell = None
        self.SIGMA_ = None
        self.initialize()

    def initialize(self):
        """initializing hidden layer and output cell"""
        self.output_cell = cu.CellUnit()
        self.output_cell.init(self.hidden_layer_size, 'x', self.weights_, self.bias_)
        self.weights_ = self.output_cell.weights_
        self.bias_ = self.output_cell.bias_
        self.SIGMA_ = []
        for i in range(self.hidden_layer_size):
            self.SIGMA_.append(np.identity(self.hidden_layer_size))
        #print(self.SIGMA_, '--------------<')

    def calc(self, x):
        return self.output_cell.calc([ac.ACTIVATION[self.rbf_func](lin.norm(x - self.T_[i]), self.SIGMA_)
                               for i in range(self.hidden_layer_size)])

    def put_data(self, X_, d_):
        """
        :param X_: input learning vectors
        :param d_: expected output vector
        """
        assert len(X_) == len(d_), 'Wrong data-set! ERR: Count of learning input vectors'
        self.X_ = X_.copy()
        self.d_ = d_.copy()

    def _green_function(self, X2_, X1_, SIGMA_):
        assert self.rbf_func in RBF_FUNCTIONS, 'ERR: Wrong radial based function'
        G = np.ndarray((len(X2_), len(X1_)+1))
        for i in range(len(X1_)):
            for j in range(len(X2_)):
                G[j][i] = (ac.ACTIVATION[self.rbf_func](lin.norm(X1_[i] - X2_[j]), SIGMA_))
        return G

    def _phi_function(self, X2_, X1_, SIGMA_):
        assert self.rbf_func in RBF_FUNCTIONS, 'ERR: Wrong radial based function'
        PHI = np.ndarray((len(X2_), len(X1_)))
        for i in range(len(X1_)):
            for j in range(len(X2_)):
                PHI[i][j] = (ac.ACTIVATION[self.rbf_func](lin.norm(X1_[i] - X2_[j]), SIGMA_))
        return PHI

    def __learn_grad(self):
        """Learning procedure. Executes learning methods up to set.
        :return LearningResult.CONVERGENCE or LearningResult.DIVERGENCE up to algorithm success"""

        assert self.X_ is not None, 'ERR: Define input vectors learning set'
        assert self.d_ is not None, 'ERR: Define output learning set'
        assert self.weights_ is not None, 'ERR: Weights are not initialized'
        assert self.bias_ is not None, 'ERR: Bias is not initialized'

        # self-organizing learning stage
        self.T_ = self._self_organize_centers(self.X_)
        print(self.T_)

        # supervised learning stage
        while sum([(self.d_[i] - self.calc(self.X_[i]))**2 for i in range(len(self.X_))]) > self.epsilon:

            self.loss_value_ = [self.d_[j] - self.calc(self.X_[j]) for j in range(len(self.X_))]
            print(self.loss_value_[3], '<---------')
            grad_wights, grad_bias, grad_center_position, grad_sigma =[], 0, [], []

            for i in range(self.hidden_layer_size):
                grad_wights.append(sum([ self.loss_value_[j] * ac.ACTIVATION[self.rbf_func](self.X_[j] - self.T_[i], self.SIGMA_[i]) for j in range(len(self.X_))]))

                grad_center_position.append(2 * self.weights_[i] * sum([self.loss_value_[j]
                    * ac.FIRST_DERIVATIVES[self.rbf_func](self.X_[j] - self.T_[i], self.SIGMA_[i]) * self.SIGMA_[i].dot(np.transpose(self.X_[j] - self.T_[i])) for j in range(len(self.X_))]))

                grad_sigma.append(- self.weights_[i] * sum([self.loss_value_[j]
                    * ac.FIRST_DERIVATIVES[self.rbf_func](self.X_[j] - self.T_[i], self.SIGMA_[i]) * self.SIGMA_[i].dot(np.transpose(self.X_[j] - self.T_[i]).dot(
                    (self.X_[j] - self.T_[i]))) for j in range(len(self.X_))]))

            #print('grad_sigma = ', grad_wights)

            grad_bias = sum([ self.loss_value_[j] for j in range(len(self.X_))])

            self.weights_ -= self.learning_rate_weight * np.array(grad_wights)
            self.bias_ -= self.learning_rate_weight * grad_bias
            self.output_cell.weights_ = self.weights_
            self.output_cell.bias_ = self.bias_
            print(self.output_cell.weights_, '<-<-')
            self.T_ -= self.learning_rate_center_position * np.array(grad_center_position)
            self.SIGMA_ = self.SIGMA_ - self.learning_rate_sigma * np.array(grad_sigma)
            print(self.weights_, self.bias_, 'T_=', self.T_, 'SIGMA_=', self.SIGMA_)
            np.random.shuffle(self.X_)

    def learn(self):
        """Learning procedure. Executes learning methods up to set.
        :return LearningResult.CONVERGENCE or LearningResult.DIVERGENCE up to algorithm success"""

        assert self.X_ is not None, 'ERR: Define input vectors learning set'
        assert self.d_ is not None, 'ERR: Define output learning set'
        assert self.weights_ is not None, 'ERR: Weights are not initialized'
        assert self.bias_ is not None, 'ERR: Bias is not initialized'

        if self.learning_method == 'sorg_green':
            self.__learn_sorg_green()
        elif self.learning_method == 'exact':
            self.__learn_exact()
        elif self.learning_method == 'grad':
            self.__learn_grad()

    def __learn_sorg_green(self):
        # self-organizing learning stage
        self.T_ = self._self_organize_centers(self.X_)
        self.SIGMA_ = self.SIGMA_[0] * (max([lin.norm(i - j) for i in self.T_ for j in self.T_]) / np.sqrt(
            2 * len(self.T_))) ** 2
        print(self._green_function(self.X_, self.T_, self.SIGMA_))
        G = self._green_function(self.X_, self.T_, self.SIGMA_)
        for i in range(len(G)):
            G[i][len(G[i]) - 1] = 1
        print(G)
        # np.append(G, np.ones(len(G)))-
        self.weights_ = np.linalg.pinv(G).dot(np.transpose(self.d_))
        self.output_cell.weights_ = self.weights_[:len(self.weights_) - 1]
        self.output_cell.bias_ = self.weights_[len(self.weights_) - 1]
        print(self.weights_)

    def __learn_exact(self):
        self.T_ = self.X_
        self.SIGMA_ = np.identity(len(self.X_))
        self.hidden_layer_size = len(self.X_)
        # print(self.SIGMA_)
        print(self._phi_function(self.X_, self.T_, self.SIGMA_))
        PHI = self._phi_function(self.X_, self.T_, self.SIGMA_)
        self.weights_ = lin.pinv(PHI).dot(np.transpose(self.d_))
        self.output_cell.weights_ = self.weights_

    def _self_organize_centers(self, X_):
        """Getting centers positions based on self-organizing"""

        # 1. initialization
        #T_ = np.random.uniform(np.min(X_), np.max(X_), (self.hidden_layer_size, self.input_size))
        T_ = np.random.uniform(0, 1.5, (self.hidden_layer_size, self.input_size))
        delta = 0
        while True:
            # 2. sampling
            x = X_[np.random.randint(0, len(X_))]

            # 3. similarity matching
            k = np.argmin([lin.norm(x - t) for t in T_])

            # 3*. end-loop condition
            if lin.norm(delta - self.learning_rate_center_position * (x - T_[k])) > self.epsilon:

                # 4. updating
                delta = self.learning_rate_center_position * (x - T_[k])
                T_[k] += self.learning_rate_center_position * (x - T_[k])

            else:
                break

        return T_

# if __name__ == '__main__':
#     a, b = dt.get_from_lab_file(dt.DEFAULT_PATH)
#     X_learn, d_learn, X_pract, d_pract = dt.split_4_to_1(a, b)
#     classifier_ = RBFNetwork(2, 15, learning_rate_weight=0.1, learning_rate_center_position=0.1, learning_rate_sigma=1.0, function='rbf_gauss')
#     classifier_.initialize()
#     # x1 = np.random.gamma(1, 3, 200)
#     # # teaching classifier
#     # classifier_.put_data([x1[i] for i in range(len(x1))], [np.sin(x1[i])**2+1 for i in range(len(x1))])
#     # print(classifier_.learn())
#     # x = np.sort([x1[i] for i in range(len(x1))])
#     #
#     # #y =  / classifier_.weights_[1]
#     # plt.scatter([i for i in x1],[0 for i in x1], color='r', s=10)
#     # plt.plot(x, [np.sin(x[i])**2+1 for i in range(len(x))], color='k', linewidth=3)
#     # plt.plot(x, [classifier_.calc(x[i]) for i in range(len(x))], color='y', linewidth=1)
#     # plt.show()
#     #
#     # for i in range(len(X_learn)):
#     #     print(classifier_.calc(X_learn[i][0]) - X_learn[i][1])
#     # teaching classifier
#
#
#     # y =  / classifier_.weights_[1]
#     X_learn = np.random.uniform(-3, 3, (100, 2))
#     d_learn = [1 if X_learn[i][0]**2 / 4 + X_learn[i][1]**2 / 9 > 1 else -1 for i in range(len(X_learn))]
#
#     classifier_.put_data(X_learn, d_learn)
#     print(classifier_.learn())
#
#     X = np.arange(-3, 3, 0.05)
#     Y = np.arange(-3, 3, 0.05)
#
#     plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) >= 0],
#                 [X_learn[i][1] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) >= 0], s=70, c='orange')
#     plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) < 0],
#                 [X_learn[i][1] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) < 0], s=70, color='red')
#
#     plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if d_learn[i] == 1],
#                 [X_learn[i][1] for i in range(len(X_learn)) if d_learn[i] == 1], s=28, color='k', marker=(5,2))
#     plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if d_learn[i] != 1],
#                 [X_learn[i][1] for i in range(len(X_learn)) if d_learn[i] != 1], s=28, c='k', marker='+')
#     Z = np.ndarray((len(X),len(Y)))
#
#     i = 0
#     for x in X:
#         j = 0
#         for y in Y:
#             print(x,y)
#             Z[j][i] = classifier_.calc(np.array([x,y]))
#             j += 1
#         i += 1
#
#     CS = plt.contour(X, Y, Z, 1, z_lim=(-0.1, 0.1), linewidths=np.arange(.5, 4, .5))
#     plt.clabel(CS, inline=1, fontsize=10)
#     plt.show()
