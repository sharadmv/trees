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
