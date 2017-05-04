import numpy as np
from tools import data_ as dt
from architectures import hopfield as hf


np.set_printoptions(threshold=np.nan, linewidth=600)
flat_image = []
for i in range(1,33):
    # reading, reversing and normalizing image
    image = dt.read_pgm("../learning_pics/" + str(i) + ".pgm", byteorder='>')
    image = dt.normalize(image)
    # flattening image matrix
    flat_image.append(image.flatten())

hopf = hf.HopfieldNet(5)
hopf.put_data(flat_image)
hopf.learn()
s = []

for i in range(32):
    s.append(hopf.calc(flat_image[i]))
print(s)


def save_weights(weights, path='../learned_weights/som2'):
    try:
        with open(path, 'w') as csvFile:
            writer = dt.csv.writer(csvFile, delimiter=';', quotechar='|')
            for i in range(len(weights)):
                writer.writerow(list(weights[i]))
    except FileNotFoundError:
        print("ERR. File is not found.")

save_weights(hopf.weights)