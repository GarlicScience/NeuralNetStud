import csv

DEFAULT_PATH = "../examples/data/data17.csv"
DEFAULT_PATH_LABS = "data/data17.csv"


def get_from_lab_file(path):
    try:
        X_, d_ = [], []
        with open(path) as csvFile:
            reader = csv.reader(csvFile, delimiter=';', quotechar='|')
            for row in reader:
                X_.append([float(f) for f in row[:len(row)-1]])
                d_.append(float(row[len(row)-1]))
        return X_, d_
    except FileNotFoundError:
        print("Error 0. File is not found.")


def split_4_to_1(X_, d_):
    l = len(d_)
    X_learn = [X_[i] for i in range(4 * l // 5)]
    X_pract = [X_[i] for i in range(4 * l // 5, l)]
    d_learn = [d_[i] for i in range(4 * l // 5)]
    d_pract = [d_[i] for i in range(4 * l // 5, l)]
    return X_learn, d_learn, X_pract, d_pract