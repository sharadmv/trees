import numpy as np

def datasets():
    return {
        'zoo',
        'awa'
    }

def load(dataset_name):
    return DATASET_LOADERS[dataset_name]()

def load_zoo():
    X = []
    y = []
    with open('data/zoo/zoo.data') as fp:
        for line in fp:
            line = line.strip().split(',')
            X.append(map(float, line[1:-1]))
            y.append(line[0])
    X = np.array(X)
    _, D = X.shape
    return X, y

def load_awa():
    DATA_DIR = "data/awa/"
    LABELS = "classes.txt"
    TRAIN = "predicate-matrix-binary.txt"
    X = np.loadtxt(DATA_DIR + TRAIN)
    animal_map = []
    with open(DATA_DIR + LABELS) as fp:
        for line in fp:
            line = line.strip()
            row = line.split()
            animal_map.append(row[1])
    return X, animal_map

DATASET_LOADERS = {
    'zoo': load_zoo,
    'awa': load_awa
}
