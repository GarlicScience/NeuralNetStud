from architectures import rbf
from matplotlib import pyplot as plt

import numpy as np


classifier_ = rbf.RBFNetwork(2, 15, learning_rate_weight=0.1, learning_rate_center_position=0.1, learning_rate_sigma=1.0,
                         function='rbf_gauss')
classifier_.initialize()



# y =  / classifier_.weights_[1]
X_learn = np.random.uniform(-3, 3, (100, 2))
d_learn = [1 if X_learn[i][0] ** 2 / 4 + X_learn[i][1] ** 2 / 9 > 1 else -1 for i in range(len(X_learn))]

classifier_.put_data(X_learn, d_learn)
print(classifier_.learn())

X = np.arange(-3, 3, 0.05)
Y = np.arange(-3, 3, 0.05)

plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) >= 0],
            [X_learn[i][1] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) >= 0], s=70, c='orange')
plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) < 0],
            [X_learn[i][1] for i in range(len(X_learn)) if classifier_.calc(X_learn[i]) < 0], s=70, color='red')

plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if d_learn[i] == 1],
            [X_learn[i][1] for i in range(len(X_learn)) if d_learn[i] == 1], s=28, color='k', marker=(5, 2))
plt.scatter([X_learn[i][0] for i in range(len(X_learn)) if d_learn[i] != 1],
            [X_learn[i][1] for i in range(len(X_learn)) if d_learn[i] != 1], s=28, c='k', marker='+')
Z = np.ndarray((len(X), len(Y)))
i = 0
for x in X:
    j = 0
    for y in Y:
        print(x, y)
        Z[j][i] = classifier_.calc(np.array([x, y]))
        j += 1
    i += 1
CS = plt.contour(X, Y, Z, 1, z_lim=(-0.1, 0.1), linewidths=np.arange(.5, 4, .5))
plt.clabel(CS, inline=1, fontsize=10)
plt.show()