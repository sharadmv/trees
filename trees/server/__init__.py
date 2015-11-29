import cPickle as pickle

from ..interact import Database, Interactor
from ..data import load
from argparse import ArgumentParser
from server import Server

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--data_dir', default='data/')
    argparser.add_argument('--subset', default=None)
    argparser.add_argument('dataset')

    return argparser.parse_args()

def main():
    import matplotlib
    matplotlib.use('Agg')
    args = parse_args()

    if args.subset is not None:
        with open(args.subset, 'rb') as fp:
            subset = pickle.load(fp)
    else:
        subset = None
    dataset = load(args.dataset)
    database = Database(args.dataset)
    interactor = Interactor(dataset, database, subset=subset)
    server = Server(interactor)
    server.initialize()
    server.listen()
