from architectures import rbf
from matplotlib import pyplot as plt
import numpy as np


classifier_ = rbf.RBFNetwork(1, 25, learning_rate_weight=0.1, learning_rate_center_position=0.1, learning_rate_sigma=1.0,
                         function='rbf_gauss')
classifier_.initialize()
x1 = np.arange(-10, 10, 0.4)
# teaching classifier
classifier_.put_data([x1[i] for i in range(len(x1))], [np.sin(x1[i]) * x1[i] for i in range(len(x1))])
print(classifier_.learn())
x = np.sort([x1[i] for i in range(len(x1))])

#y =  / classifier_.weights_[1]
plt.scatter([i for i in x1],[0 for i in x1], color='r', s=10)
plt.plot(x, [np.sin(x[i]) * x[i] for i in range(len(x))], color='k', linewidth=3)
plt.plot(x, [classifier_.calc(x[i]) for i in range(len(x))], color='y', linewidth=1)
plt.axes([-10, 10, -10, 10])

fig = plt.figure()
classifier_2 = rbf.RBFNetwork(1, 20, learning_rate_weight=0.1, learning_rate_center_position=0.1, learning_rate_sigma=1.0,
                         function='rbf_gauss')
classifier_2.initialize()
x1 = np.arange(-8, 8, 0.4)
# teaching classifier
classifier_2.put_data([x1[i] for i in range(len(x1))], [np.arctan(x1[i]) for i in range(len(x1))])
print(classifier_2.learn())
x = np.sort([x1[i] for i in range(len(x1))])

plt.scatter([i for i in x1], [0 for i in x1], color='r', s=10)
plt.plot(x, [np.arctan(x[i]) for i in range(len(x))], color='k', linewidth=3)
plt.plot(x, [classifier_2.calc(x[i]) for i in range(len(x))], color='y', linewidth=1)
plt.axes([-8, 8, -8, 8])


plt.figure()
classifier_3 = rbf.RBFNetwork(1, 0, learning_method='exact', learning_rate_weight=0.1, learning_rate_center_position=0.1, learning_rate_sigma=1.0,
                         function='rbf_gauss')
classifier_3.initialize()
x1 = np.arange(-4, 4, 0.37)
# teaching classifier
classifier_3.put_data([x1[i] for i in range(len(x1))], [abs(np.arctan(1 - x1[i])) + np.sin(x1[i]) for i in range(len(x1))])
print(classifier_3.learn())
x = np.sort([x1[i] for i in range(len(x1))])

plt.scatter([i for i in x1], [0 for i in x1], color='r', s=10)
plt.plot(x, [abs(np.arctan(1 - x[i])) + np.sin(x[i]) for i in range(len(x))], color='k', linewidth=3)
plt.plot(x, [classifier_3.calc(x[i]) for i in range(len(x))], color='y', linewidth=1)
plt.axes([-4, 4, -4, 4])
plt.show()
