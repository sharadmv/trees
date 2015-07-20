import matplotlib.pyplot as plt
import seaborn as sns
sns.despine()
import numpy as np
import networkx as nx

def plot_tssb(tssb):
    g = nx.DiGraph()
    assert tssb.root is not None

    add_nodes(g, tssb.root)

    pos = nx.graphviz_layout(g, prog='dot', args='-Granksep=100.0')
    labels = {n: n.point_count for n in g.nodes()}
    nx.draw_networkx_nodes(g, pos,
                            node_color='b',
                            node_size=300,
                            alpha=0.8)
    nx.draw_networkx_edges(g, pos,
                            alpha=0.8, arrows=False)
    nx.draw_networkx_labels(g, pos, labels, font_size=12, font_color='w')

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
    color_map = sns.light_palette("purple", n_colors=len(nodes), reverse=True)
    colors = {}
    for i, n in enumerate(sorted(nodes, key=lambda x: len(x))):
        colors[n] = color_map[i]
    for i, (x, y) in enumerate(X):
        plt.scatter(x, y, color=colors[z[i]])
    if tssb is not None:
        for node in tssb.dfs():
            #if node in nodes:
            (x, y) = node.parameter
            plt.scatter(x, y, color='r', alpha=0.2)
            if node.parent is not None:
                (x2, y2) = node.parent.parameter
                plt.plot([x2, x], [y2, y], color='r', alpha=0.1)
