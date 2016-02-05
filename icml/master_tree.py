import numpy as np
from trees import Tree, TreeNode, TreeLeaf
import cPickle as pickle

def make_master(X, y, name):
    if name == 'mnist' or name == 'iris' or name == '20news':
        return make_class_tree(X, y)
    if name == 'zoo':
        with open('data/zoo/zoo2.tree', 'rb') as fp:
            tree = pickle.load(fp)
        return tree

def make_class_tree(X, y):
    tree = Tree()
    tree.root = TreeNode()
    C = np.unique(y)
    nodes = {c: TreeNode() for c in C}
    for i, c in enumerate(y):
        node = nodes[c]
        leaf = TreeLeaf(i)
        node.add_child(leaf)
    for node in nodes.values():
        tree.root.add_child(node)
    return tree

if __name__ == "__main__":
    from trees.data import load
    from util import plot_tree
    mnist = load('mnist')
    X, y = mnist.X, mnist.y.astype(np.int)
    idx = np.random.permutation(xrange(X.shape[0]))[:100]
    X = X[idx]
    y = y[idx]
    tree = make_master(X, y, 'mnist')
    plot_tree(tree)
