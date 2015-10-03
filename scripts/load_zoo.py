import cPickle as pickle
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('tree_path')

args = argparser.parse_args()
print args

with open(args.tree_path, 'rb') as fp:
    tree = pickle.load(fp)
