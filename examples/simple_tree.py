from trees.util import plot_tree
from trees import Tree, TreeNode, TreeLeaf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    leaf1 = TreeLeaf(1)
    leaf2 = TreeLeaf(2)
    leaf3 = TreeLeaf(3)
    node1 = TreeNode()
    node1.add_child(leaf1)
    node1.add_child(leaf2)
    node2 = TreeNode()
    node2.add_child(node1)
    node2.add_child(leaf3)
    tree = Tree(node2)

    plt.figure()
    plot_tree(tree)
    plt.show()

    p = leaf1.detach()
    plt.figure()
    plot_tree(tree)
    plt.show()

    leaf3.attach(p)
    plt.figure()
    plot_tree(tree)
    plt.show()
