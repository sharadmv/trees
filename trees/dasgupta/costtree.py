import numpy as np
from .. import Tree

class DasguptaTree(Tree):

    def __init__(self, D, T, *args, **kwargs):
        self.D = D
        self.T = float(T)
        super(DasguptaTree, self).__init__(*args, **kwargs)

    def cost(self, node=None):
        node = node or self.root
        if node.is_leaf():
            return 0
        left_points = node.children[0].points()
        right_points = node.children[1].points()

        c = sum([self.D[left_point][right_point] for left_point in left_points for right_point in right_points])
        return node.leaf_count() * c + sum([self.cost(n) for n in node.children])

    def marg_log_likelihood(self):
        return -self.cost() / self.T

    def copy(self):
        tree = self.__class__(self.D, self.T, root=self.root.copy(),
                              constraints=self.constraints,
                              **self.parameters.copy())
        return tree

    def log_prob_assignment(self, assignment, node=None):
        return 1.0 / (self.root.leaf_count() - 1)

    def sample_assignment(self, **kwargs):
        choice = self.choice(ignore=False)
        return (choice.get_index(), ()), 1.0 / (self.root.leaf_count() - 1)

    def assign_node(self, node, assignment):
        (idx, state) = assignment
        assignee = self.get_node(idx)
        assignee.attach(node)
