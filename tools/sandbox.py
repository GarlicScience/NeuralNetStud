import numpy as np
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
T_ = np.random.uniform(0, 1, (3, 2))
print(T_)
T_[0]+= [2, 3]
print(T_)

b = np.array([1, 2])