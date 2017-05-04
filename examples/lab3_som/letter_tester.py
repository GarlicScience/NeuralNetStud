import tools.data_ as dt
import numpy as np
from matplotlib import pyplot

from architectures import self_organizing_maps as som


def get_weights(path='../learned_weights/som1'):
    try:
        with open(path, 'r') as csvFile:
            reader = dt.csv.reader(csvFile, delimiter=';', quotechar='|')
            w_ = []
            for row in reader:
                w_.append(np.array([float(i) for i in row]))
            return w_

    except FileNotFoundError:
        print("ERR. File is not found.")


def indexize(s):
    return [dt.LETTERS_RU[l]+1 for l in s]

w = get_weights()

flat_image = []
for i in indexize('АБЕВЕГЕДЕЙКА'):
    # reading, reversing and normalizing image
    image = dt.read_pgm("learning_pics/" + str(i) + ".pgm", byteorder='>')
    image = dt.normalize(image)
    # flattening image matrix
    flat_image.append(image.flatten())

s_map = som.KohonenMap(32, len(flat_image[0]), sigma0=3, weights=w)

for i in range(len('АБЕВЕГЕДЕЙКА')):
    k = s_map.calc(flat_image[i])
    leng = int(len(w[k])**0.5)
    img = np.ndarray((leng, leng))
    for i1 in range(leng):
        for j1 in range(leng):
            img[i1][j1] = w[k][i1 * leng + j1]
    pyplot.imshow(img, pyplot.cm.gray)
    pyplot.show()
    print(k)

#  А - 2,   Б - 29,   В - 28,   Г - 3,
#  Д - 4,   Е - 29,   Ж - 30,   З - 27,
#  И - 11,  Й - 11,   К - 23,   Л - 9,
#  М - 18,  Н - 11,   О - 26,   П - 10,
#  Р - 22,  С - 26,   Т - 2,    У - 13,
#  Ф - 31,  Х - 14,   Ц - 5,    Ч - 15,
#  Ш - 0,   Щ - 0,    Ъ - 12,   Ы - 24,
#  Ь - 17,  Э - 27,   Ю - 6,    Я - 15.