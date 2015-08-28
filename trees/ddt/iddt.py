from ddt import DirichletDiffusionTree
from inode import InteractiveNode

class InteractiveDirichletDiffusionTree(DirichletDiffusionTree):

    def __init__(self, df, likelihood_model, constraints=[]):
        super(InteractiveDirichletDiffusionTree, self).__init__(df, likelihood_model)
        self.constraints = constraints

    def initialize_assignments(self, X):
        N, _ = X.shape
        self.root = InteractiveNode.construct(set(xrange(N)), self.constraints, X)
        self.root.time = 0
        self.root.state = self.likelihood_model.mu0

    def sample_assignment(self, points=None):
        return self.root.sample_assignment(self.df, self.constraints, points)

    def add_constraint(self, constraint, X):
        self.constraints.append(constraint)
        a, b, c = constraint
        node = self.mrca(a, b, c)
        points = node.points()
        new_node = InteractiveNode.construct(points, self.constraints, X)
        if node.parent is None:
            self.root = new_node
        else:
            node.parent.replace_child(node, new_node)
            node.parent = None

    def copy(self):
        ddt = InteractiveDirichletDiffusionTree(self.df, self.likelihood_model, constraints=self.constraints)
        ddt.root = self.root.copy()
        return ddt

    def verify_constraint(self, constraint):
        (a, b, c) = constraint
        _, (ai, _) = self.point_index(a)
        _, (bi, _) = self.point_index(b)
        _, (ci, _) = self.point_index(c)

        while True:
            if ai[0] == bi[0]:
                if ci[0] != ai[0]:
                    return True
                ai = ai[1:]
                bi = bi[1:]
                ci = ci[1:]
            if ai[0] != bi[0]:
                return False

    def verify_constraints(self, constraints):
        return all(self.verify_constraint(c) for c in constraints)
