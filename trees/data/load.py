import numpy as np

from ..tssb import GaussianParameterProcess

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
    mu0 = np.zeros(D)
    sigma0 = np.eye(D) * 0.001
    sigmat = np.eye(D) * 0.002
    sigma = np.eye(D) * 0.01
    p = GaussianParameterProcess(mu0, sigma0, sigmat, sigma)
    X = np.log(X + 0.1)
    return X, y, p

DATASET_LOADERS = {
    'zoo': load_zoo
}
