from path import Path
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt, mpld3

class Dataset(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def convert(self, x, y):
        raise NotImplementedError

def datasets():
    return {
        'zoo',
        'awa',
        'mnist'
    }

class AnimalDataset(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def convert(self, i):
        return self.y[i]

class ImageDataset(object):

    def __init__(self, X, y, size=None):
        self.X = X
        self.y = y
        self.size = size

    def convert(self, i):
        fig, ax = plt.subplots()
        fig.set_figheight(1)
        fig.set_figwidth(1)
        ax.imshow(self.X[i].reshape(self.size))
        return mpld3.fig_to_html(fig)

def load(dataset_name, data_dir='data/'):
    data_dir = Path(data_dir)
    return DATASET_LOADERS[dataset_name](data_dir)

def load_zoo(data_dir):
    X = []
    y = []
    removed_frog = False
    with open(data_dir / 'zoo/zoo.data') as fp:
        for line in fp:
            line = line.strip().split(',')
            if line[0] in ('goat', 'oryx', 'toad'):
                continue
            if not removed_frog and line[0] == 'frog':
                removed_frog = True
                continue
            X.append(map(float, line[1:-1]))
            y.append(line[0])
    X = np.array(X)
    _, D = X.shape
    return AnimalDataset(X, y)

def load_awa(data_dir):
    DATA_DIR = data_dir / "awa"
    LABELS = "classes.txt"
    TRAIN = "predicate-matrix-binary.txt"
    X = np.loadtxt(DATA_DIR + TRAIN)
    animal_map = []
    with open(DATA_DIR + LABELS) as fp:
        for line in fp:
            line = line.strip()
            row = line.split()
            animal_map.append(row[1])
    return AnimalDataset(X, animal_map)

def load_mnist(data_dir):
    DATA_DIR = data_dir / "mnist"
    mnist = fetch_mldata('MNIST original', data_home=DATA_DIR)
    X, y = mnist['data'], mnist['target']
    return ImageDataset(X, y, (28, 28))

DATASET_LOADERS = {
    'zoo': load_zoo,
    'awa': load_awa,
    'mnist': load_mnist
}
