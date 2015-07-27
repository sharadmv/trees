import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.despine()
import numpy as np
import networkx as nx

def plot_tssb(tssb, ax=None):
    g = nx.DiGraph()
    assert tssb.root is not None

    add_nodes(g, tssb.root)

    pos = nx.graphviz_layout(g, prog='dot', args='-Granksep=100.0')
    labels = {n: n.point_count for n in g.nodes()}
    nx.draw_networkx_nodes(g, pos,
                            node_color='b',
                            node_size=300,
                            alpha=0.8, ax=ax)
    nx.draw_networkx_edges(g, pos,
                            alpha=0.8, arrows=False, ax=ax)
    nx.draw_networkx_labels(g, pos, labels, font_size=12, font_color='w', ax=ax)

def add_nodes(g, node):
    for c, child_node in node.children.items():
        g.add_edge(node, child_node)
        add_nodes(g, child_node)

def generate_data(N, tssb, collect=True):
    data = []
    y = []
    for i in xrange(N):
        node, index = tssb.sample_one()
        data.append(node.sample_one())
        y.append(index)
    if collect:
        tssb.garbage_collect()
    return np.array(data), y

def plot_data(X, z, tssb=None):
    nodes = set(z)
    color_map = sns.color_palette("coolwarm", len(set(map(len, nodes))))
    colors = {}
    for c, n in zip(color_map, set(map(len, nodes))):
        colors[n] = c
    for i, (x, y) in enumerate(X):
        plt.scatter(x, y, color=colors[len(z[i])])

def save_tssb(tssb, location):
    with open(location, 'wb') as fp:
        pickle.dump(tssb.get_state(), fp)

def load_tssb(location):
    with open(location, 'rb') as fp:
        tssb = pickle.load(fp)
    return tssb
