import logging
import mpld3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from node import Leaf
from .. import Tree

class DirichletDiffusionTree(Tree):

    def __init__(self, df, likelihood_model):
        self.df = df
        self.likelihood_model = likelihood_model
        self.root = None

    def copy(self):
        ddt = DirichletDiffusionTree(self.df, self.likelihood_model)
        ddt.root = self.root.copy()
        return ddt

    def marg_log_likelihood(self):
        assert self.root is not None
        _, tree_structure, data_structure = self.root.log_likelihood(self.df, self.likelihood_model, self.root.time)
        return tree_structure + data_structure

    def dfs(self):
        assert self.root is not None
        yield self.root
        s = set(self.root.children)
        while len(s) > 0:
            child = s.pop()
            yield child
            if not isinstance(child, Leaf):
                s.update(child.children)

    def get_node(self, index):
        return self.root.get_node(index)

    def point_index(self, point):
        return self.root.point_index(point)

    def uniform_index(self, u, ignore_depth=0):
        return self.root.uniform_index(u, ignore_depth=ignore_depth)

    def sample_assignment(self):
        return self.root.sample_assignment(self.df)

    def log_prob_assignment(self, assignment):
        return self.root.log_prob_assignment(self.df, assignment)

    def choice(self):
        return self.uniform_index(np.random.random())

    def update_latent(self):
        self.root.update_latent(self.likelihood_model)

    def __getitem__(self, key):
        return self.root[key]

    def plot_mpld3(self, y):
        fig, ax = plt.subplots(1, 1)
        g, nodes, node_labels = self.plot(ax=ax)
        labels = []
        for node in g.nodes_iter():
            if isinstance(node, Leaf):
                labels.append("<div class='tree-label'><div class='tree-label-text'>%s</div></div>"
                              % self.y[node.point])
            else:
                labels.append("<div class='tree-label'><div class='tree-label-text'>%f</div></div>"
                              % node.time)
        tooltip = mpld3.plugins.PointHTMLTooltip(nodes, labels=labels)
        mpld3.plugins.connect(fig, tooltip)
        tooltip = mpld3.plugins.PointHTMLTooltip(node_labels, labels=labels)
        mpld3.plugins.connect(fig, tooltip)
        plt.axis('off')
        return fig

    def plot(self, ax=None):
        g = nx.DiGraph()
        assert self.root is not None

        def add_nodes(node):
            if not isinstance(node, Leaf):
                for child in node.children:
                    g.add_edge(node, child)
                    add_nodes(child)

        add_nodes(self.root)

        pos = nx.graphviz_layout(g, prog='dot', args='-Granksep=100.0')
        labels = {n: n.point_count() if isinstance(n, Leaf) else "" for n in g.nodes()}
        node_size = [120 if isinstance(n, Leaf) else 40 for n in g.nodes()]
        nodes = nx.draw_networkx_nodes(g, pos,
                               node_color='b',
                               node_size=node_size,
                               alpha=0.8, ax=ax)
        nx.draw_networkx_edges(g, pos,
                                alpha=0.8, arrows=False, ax=ax)
        labels = nx.draw_networkx_labels(g, pos, labels, font_size=10, font_color='w', ax=ax)
        return g, nodes, labels
