"""
Rosenblatt perceptron classifier example. (AKA 'Lab 1')
Data-set variant is '17'
"""
import numpy as np
from architectures import perceptron as pt
from tools import data_ as dt
import pylab as plt


def save_weights(weights, bias, path='learned_weights/perceptron17'):
    try:
        with open(path, 'a') as csvFile:
            writer = dt.csv.writer(csvFile, delimiter=';', quotechar='|')
            weights = list(weights)
            weights.append(bias)
            writer.writerow(weights)
    except FileNotFoundError:
        print("ERR. File is not found.")


def get_weights(path='learned_weights/perceptron17'):
    try:
        with open(path, 'r') as csvFile:
            reader= dt.csv.reader(csvFile, delimiter=';', quotechar='|')
            for row in reader:
                w_ = np.array([float(i) for i in row[:len(row) - 1]])
                b_ = float(row[len(row) - 1])
                return w_, b_

    except FileNotFoundError:
        print("ERR. File is not found.")


# loading saved weights
w = get_weights()

# draining data-set
a, b = dt.get_from_lab_file(dt.DEFAULT_PATH_LABS)
X_learn, d_learn, X_pract, d_pract = dt.split_4_to_1(a, b)

# initializing classifier
classifier_ = pt.RosenblattClassifier(2, function='sgn', learning_rate=0.001)
classifier_.set_log_(True)

# teaching classifier
classifier_.put_data(X_learn, d_learn)
print(classifier_.learn())

# saving weights
save_weights(classifier_.weights_, classifier_.bias_)

X_check = []
for i in range(40):
    X_check.append(np.random.uniform(0,1,2))

for i in range(len(X_check)):
    print(X_check[i][0], X_check[i][1], classifier_.calc(X_check[i]))

# plotting convergence diagram
# plt.plot(classifier_.ep_, classifier_.l_)
# plt.show()

# visualizing 2d case!
# plt.scatter([a[i][0] for i in range(len(a)) if b[i] != 1],
#             [a[i][1] for i in range(len(a)) if b[i] != 1], s=2)
# plt.scatter([a[i][0] for i in range(len(a)) if b[i] == 1],
#             [a[i][1] for i in range(len(a)) if b[i] == 1], s=2)

plt.scatter([X_check[i][0] for i in range(len(X_check)) if classifier_.calc(X_check[i]) != 1],
            [X_check[i][1] for i in range(len(X_check)) if classifier_.calc(X_check[i]) != 1], s=2)
plt.scatter([X_check[i][0] for i in range(len(X_check)) if classifier_.calc(X_check[i]) == 1],
            [X_check[i][1] for i in range(len(X_check)) if classifier_.calc(X_check[i]) == 1], s=2)
x = np.linspace(0, 1)
y = (-classifier_.weights_[0] * x - classifier_.bias_) / classifier_.weights_[1]
plt.plot(x,y,color='k',linewidth=1)
plt.show()
