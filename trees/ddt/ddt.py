import logging
import mpld3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from node import NonTerminal, Leaf

class DirichletDiffusionTree(object):

    def __init__(self, X, y, df, likelihood_model):
        self.X = X
        self.y = y
        self.df = df
        self.likelihood_model = likelihood_model

        self.N, self.D = self.X.shape
        self.root = None
        self.initialize_assignments()

    def initialize_assignments(self):
        nodes = [Leaf(None, i, self.X[i]) for i in xrange(self.N)]
        while len(nodes) > 1:
            random.shuffle(nodes)
            (a, b), rest = nodes[:2], nodes[2:]
            merge_time = random.uniform(0, min(a.time, b.time))
            node = NonTerminal(a, b, None, (a.state + b.state) / 2.0, merge_time)
            a.parent = node
            b.parent = node
            nodes = [node] + rest
        self.root = nodes[0]
        self.root.time = 0
        self.root.state = self.likelihood_model.mu0

    def copy(self):
        ddt = DirichletDiffusionTree(self.X, self.y, self.df, self.likelihood_model)
        ddt.root = self.root.copy()
        return ddt

    def log_likelihood(self):
        assert self.root is not None
        _, tree_structure, data_structure = self.root.log_likelihood(self.df, self.likelihood_model, self.root.time)
        return tree_structure + data_structure

    def point_index(self, point):
        return self.root.point_index(point)
    def choice(self):
        nodes = self.root.nodes()
        nodes.remove(self.root)
        new_nodes = []
        for node in nodes:
            if not node.parent == self.root:
                new_nodes.append(node)
        return random.choice(new_nodes)

    def remove_parent(self, node):
        assert node is not self.root, "Root has no parent."
        if node.parent == self.root:
            old = self.root
            self.root = None
            return old, None, 0.0

        parent = node.parent
        grandparent = parent.parent

        parent_index = parent.index(node)
        grandparent_index = grandparent.index(parent)
        other_parent = grandparent.get_child(1 - grandparent_index)
        other_child = parent.get_child(1 - parent_index)
        other_child_count = other_child.point_count()

        side_probability = np.log(other_child_count) - np.log(other_child_count +
                                                              node.point_count() +
                                                              other_parent.point_count())
        time_probability = -np.log(parent.time - grandparent.time)

        grandparent.set_child(grandparent_index, parent.get_child(1 - parent_index))
        parent.get_child(1 - parent_index).parent = grandparent

        parent.children.remove(parent.get_child(1 - parent_index))

        return parent, grandparent, side_probability + time_probability

    def attach_node(self, node, parent):
        if parent is None:
            self.root = node
            return 0.0
        num_points = parent.point_count()
        probs = [parent.left.point_count() / float(num_points), parent.right.point_count() / float(num_points)]
        logging.debug("Choosing path for node: %s" % str(probs))
        choice = np.random.choice([0, 1], p=probs)
        side_probability = np.log(probs[choice])
        logging.debug("Chose path: %u (%f)" % (choice, probs[choice]))

        old_node = parent.get_child(choice)
        old_node.parent = node
        node.children.append(old_node)

        parent.set_child(choice, node)
        node.parent = parent


        node.time = random.uniform(node.parent.time, min(node.left.time, node.right.time))
        time_probability = -np.log(min(node.left.time, node.right.time) - node.parent.time)

        return side_probability + time_probability

    def update_latent(self):
        self.root.update_latent(self.likelihood_model)

    def __getitem__(self, key):
        return self.root[key]

    def plot_mpld3(self):
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
