import numpy as np
from tools import data_ as dt
from architectures import self_organizing_maps as som


np.set_printoptions(threshold=np.nan, linewidth=600)
flat_image = []
for i in range(1,33):
    # reading, reversing and normalizing image
    image = dt.read_pgm("learning_pics/" + str(i) + ".pgm", byteorder='>')
    image = dt.normalize(image)
    # flattening image matrix
    flat_image.append(image.flatten())

s_map = som.KohonenMap(32, len(flat_image[0]), sigma0=3, learning_rate0=0.1)
s_map.put_data(flat_image * 10000)
s_map.learn()
s = []

for i in range(32):
    s.append(s_map.calc(flat_image[i]))
print(s)


def save_weights(weights, path='../learned_weights/som2'):
    try:
        with open(path, 'w') as csvFile:
            writer = dt.csv.writer(csvFile, delimiter=';', quotechar='|')
            for i in range(len(weights)):
                writer.writerow(list(weights[i]))
    except FileNotFoundError:
        print("ERR. File is not found.")

save_weights(s_map.weights)