import matplotlib.pyplot as plt
import networkx as nx

def plot_tree(tree, mpld3=False, ax=None, y=None):
    g = nx.DiGraph()
    assert tree.root is not None

    def add_nodes(node):
        if not node.is_leaf():
            for child in node.children:
                g.add_edge(node, child)
                add_nodes(child)

    add_nodes(tree.root)

    pos = nx.graphviz_layout(g, prog='dot', args='-Granksep=100.0')
    labels = {n: tree.node_as_string(n) for n in g.nodes()}
    node_size = [120 if n.is_leaf() else 40 for n in g.nodes()]
    nodes = nx.draw_networkx_nodes(g, pos,
                            node_color='b',
                            node_size=node_size,
                            alpha=0.8, ax=ax)
    nx.draw_networkx_edges(g, pos,
                            alpha=0.8, arrows=False, ax=ax)
    node_labels = nx.draw_networkx_labels(g, pos, labels, font_size=10, font_color='r', ax=ax)
    if mpld3:
        fig, ax = plt.subplots(1, 1)
        labels = []
        for node in g.nodes_iter():
            if node.is_leaf():
                labels.append("<div class='tree-label'><div class='tree-label-text'>%s</div></div>"
                              % y[node.point])
            else:
                labels.append("<div class='tree-label'><div class='tree-label-text'>%f, %s</div></div>"
                              % str(node.state))
        tooltip = mpld3.plugins.PointHTMLTooltip(nodes, labels=labels)
        mpld3.plugins.connect(fig, tooltip)
        tooltip = mpld3.plugins.PointHTMLTooltip(node_labels, labels=labels)
        mpld3.plugins.connect(fig, tooltip)
        plt.axis('off')
        return fig

def plot_tree_2d(tree, X):
    plt.scatter(*X.T)
    def plot_node(node, size=40):
        if node.is_leaf():
            return
        plt.scatter(*node.get_state('latent_value'), color='g', alpha=0.5, s=size)
        for child in node.children:
            plt.plot(*zip(node.get_state('latent_value'), child.get_state('latent_value')), color='g', alpha=0.2)
            plot_node(child, size=size/2.0)
    plot_node(tree.root)
