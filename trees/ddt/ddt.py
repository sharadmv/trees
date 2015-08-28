import logging
import mpld3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from node import Node
from .. import Tree

class DirichletDiffusionTree(Tree):

    def __init__(self, df, likelihood_model):
        self.df = df
        self.likelihood_model = likelihood_model
        self.root = None
        self._marg_log_likelihood = None

    def initialize_assignments(self, X):
        N, _ = X.shape
        self.root = Node.construct(set(xrange(N)), X)
        self.root.time = 0.0
        self.root.state = self.likelihood_model.mu0

    def copy(self):
        ddt = DirichletDiffusionTree(self.df, self.likelihood_model)
        ddt.root = self.root.copy()
        return ddt

    def initialize_assignments(self, X):
        N, _ = X.shape
        self.root = Node.construct(set(xrange(N)), X)
        self.root.time = 0
        self.root.state = self.likelihood_model.mu0

    def marg_log_likelihood(self):
        if self._marg_log_likelihood is None:
            assert self.root is not None
            _, tree_structure, data_structure = self.root.log_likelihood(self.df, self.likelihood_model, self.root.time)
            self._marg_log_likelihood = tree_structure + data_structure
        return self._marg_log_likelihood

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

    def attach_node(self, node, assignment):
        return self.root.attach_node(node, assignment)

    def point_index(self, point):
        return self.root.point_index(point)

    def uniform_index(self, u, ignore_depth=0):
        return self.root.uniform_index(u, ignore_depth=ignore_depth)

    def sample_assignment(self, points=None):
        return self.root.sample_assignment(self.df)

    def log_prob_assignment(self, assignment):
        return self.root.log_prob_assignment(self.df, assignment)

    def choice(self, ignore_depth=0):
        return self.uniform_index(np.random.random(), ignore_depth=ignore_depth)

    def update_latent(self):
        self.root.update_latent(self.likelihood_model)

    def mrca(self, a, b, c):
        a_idx = self.point_index(a)[1][0]
        b_idx = self.point_index(b)[1][0]
        c_idx = self.point_index(c)[1][0]
        idx = ()
        i = 0
        while True:
            if a_idx[i] == b_idx[i] and a_idx[i] == c_idx[i]:
                idx += (a_idx[i],)
                i += 1
            else:
                return self[idx]

    def __getitem__(self, key):
        return self.root[key]

    def plot_mpld3(self, y):
        fig, ax = plt.subplots(1, 1)
        g, nodes, node_labels = self.plot(ax=ax)
        labels = []
        for node in g.nodes_iter():
            if node.is_leaf():
                labels.append("<div class='tree-label'><div class='tree-label-text'>%s</div></div>"
                              % y[node.point])
            else:
                labels.append("<div class='tree-label'><div class='tree-label-text'>%f, %s</div></div>"
                              % (node.time, str(node.state)))
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
            if not node.is_leaf():
                for child in node.children:
                    g.add_edge(node, child)
                    add_nodes(child)

        add_nodes(self.root)

        pos = nx.graphviz_layout(g, prog='dot', args='-Granksep=100.0')
        labels = {n: n.point_count() if n.is_leaf() else "" for n in g.nodes()}
        node_size = [120 if n.is_leaf() else 40 for n in g.nodes()]
        nodes = nx.draw_networkx_nodes(g, pos,
                               node_color='b',
                               node_size=node_size,
                               alpha=0.8, ax=ax)
        nx.draw_networkx_edges(g, pos,
                                alpha=0.8, arrows=False, ax=ax)
        labels = nx.draw_networkx_labels(g, pos, labels, font_size=10, font_color='w', ax=ax)
        return g, nodes, labels
