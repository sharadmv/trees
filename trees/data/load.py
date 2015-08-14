import numpy as np

def datasets():
    return {
        'zoo'
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

DATASET_LOADERS = {
    'zoo': load_zoo
}
