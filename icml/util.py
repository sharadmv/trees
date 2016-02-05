from cStringIO import StringIO
import matplotlib.pyplot as plt
from Bio import Phylo

def plot_tree(tree, y):
    final_tree = tree.copy()
    for node in final_tree.dfs():
        if node.is_leaf():
            node.point = y[node.point]

    newick = final_tree.to_newick()
    tree = Phylo.read(StringIO(newick), 'newick')
    Phylo.draw_graphviz(tree, prog='neato')

def plot_tree_2d(tree, X, pca=None):
    if pca is None:
        plt.scatter(*X.T)
    else:
        plt.scatter(*pca.transform(X).T)
    def plot_node(node, size=40):
        if node.is_leaf():
            return
        if pca is not None:
            lv = pca.transform(node.get_state('latent_value')).ravel()
        else:
            lv = node.get_state('latent_value')
        plt.scatter(*lv, color='g', alpha=0.5, s=size)
        for child in node.children:
            if pca is not None:
                clv = pca.transform(child.get_state('latent_value')).ravel()
            else:
                clv = child.get_state('latent_value')
            plt.plot(*zip(lv, clv), color='g', alpha=0.2)
            plot_node(child, size=size/2.0)
    plot_node(tree.root)

def dist(master_tree, tree):
    all_constraints = master_tree.generate_constraints()
    d = 0
    for constraint in all_constraints:
        d += not tree.verify_constraint(constraint)
    return d / float(len(all_constraints))

