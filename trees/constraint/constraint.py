from tqdm import tqdm
import numpy as np
import random

class ConstraintGetter(object):

    def __init__(self, master_tree, y, classification=False):
        self.master_tree = master_tree
        self.y = y
        self.classification = classification

    def get_constraint(self, trees):
        raise NotImplementedError

    def violated_constraint(self, constraint):
        if not self.classification:
            (a, b, c) = constraint
            if self.master_tree.verify_constraint((a, b, c)):
                return False
            if self.master_tree.verify_constraint((c, b, a)):
                return (c, b, a)
            return (c, a, b)
        ai, bi, ci = constraint
        a, b, c = [self.y[i] for i in constraint]
        if a == b:
            return False
        if a == c:
            return (ai, ci, bi)
        if b == c:
            return (bi, ci, ai)
        return False

    def pick_constraint(self, tree):
        def pick(tree):
            assert len(tree.children) == 2
            left, right = tree.children[0], tree.children[1]
            left_points, right_points = left.points(), right.points()
            for i in left_points:
                for j in left_points:
                    if i != j:
                        for k in right_points:
                            result = self.violated_constraint((i, j, k))
                            if result is not False:
                                return result
            for i in right_points:
                for j in right_points:
                    if i != j:
                        for k in left_points:
                            result = self.violated_constraint((i, j, k))
                            if result is not False:
                                return result
            result = None
            if result is None and not left.is_leaf():
                result = pick(left)
            if result is None and not right.is_leaf():
                result = pick(right)
            return result
        return pick(tree.root)

class TotallyRandom(ConstraintGetter):

    def __init__(self, master_tree, y, **kwargs):
        super(TotallyRandom, self).__init__(master_tree, y, **kwargs)
        self.satisfied = set()
        self.all_constraints = list(self.master_tree.generate_constraints())

    def get_constraint(self, _):
        constraint = random.choice(self.all_constraints)
        while constraint in self.satisfied:
            constraint = random.choice(self.all_constraints)
        self.satisfied.add(constraint)
        return constraint

class SmartRandom(ConstraintGetter):

    def __init__(self, master_tree, y, **kwargs):
        super(SmartRandom, self).__init__(master_tree, y, **kwargs)

    def get_constraint(self, trees):
        for tree in trees:
            constraint = self.pick_constraint(tree)
            if constraint is not None:
                return constraint

class StupidRandom(ConstraintGetter):

    def __init__(self, master_tree, y, K=10, **kwargs):
        super(StupidRandom, self).__init__(master_tree, y, **kwargs)
        self.K = K

    def get_constraint(self, trees):
        points = trees[-1].points()
        idx = list(random.sample(points, self.K))
        for tree in trees:
            tree = tree.induced_subtree(idx)
            constraint = self.pick_constraint(tree)
            if constraint is not None:
                return constraint


class Variance(SmartRandom):

    def __init__(self, master_tree, y, N=10, K=10, **kwargs):
        super(Variance, self).__init__(master_tree, y, **kwargs)
        self.N = N
        self.K = K
        self.points = self.master_tree.points()

    def get_vars(self, trees):
        sub_idx = []
        depths = []
        for i in tqdm(xrange(self.N), nested=True):
            idx = random.sample(self.points, self.K)
            sub_idx.append(idx)
            subtree = []
            depth = []
            for t in tqdm(trees, nested=True):
                st = t.induced_subtree(idx)
                subtree.append(st)
                depth.append(create_depth_matrix(st))
            depths.append(depth)
        depths = np.array(depths)
        std = depths.std(axis=1)
        vars = []
        triu = np.triu_indices(self.K)
        for i in xrange(self.N):
            vars.append(std[i][triu].max())
        return np.array(vars), np.array(sub_idx)

    def get_constraint(self, trees):
        vars, idx = self.get_vars(trees)
        for i in vars.argsort()[::-1]:
            constraint = self.pick_constraint(trees[-1].induced_subtree(idx[i]))
            if constraint is not None:
                return constraint

class Hybrid(Variance):

    def __init__(self, master_tree, y, N=10, K=10, **kwargs):
        super(Hybrid, self).__init__(master_tree, y, **kwargs)
        self.N = N
        self.K = K
        self.points = self.master_tree.points()
        self.query_state = True

    def get_constraint(self, trees):
        if self.query_state:
            constraint = Variance.get_constraint(self, trees)
        else:
            constraint = SmartRandom.get_constraint(self, trees)
        self.query_state = not self.query_state
        return constraint

def get_tree_distance(u, v):
    i = 0
    if u == v:
        return 0
    while u[i] == v[i]:
        i += 1
    return len(u[i:]) + len(v[i:])

def create_depth_matrix(tree):
    points = list(tree.points())
    N = len(points)
    mat = np.zeros((N, N))
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            u, v = tree.point_index(p1), tree.point_index(p2)
            if u is None or v is None:
                mat[i, j] = np.inf
                mat[j, i] = np.inf
            else:
                mat[i, j] = get_tree_distance(u, v)
                mat[j, i] = get_tree_distance(u, v)
    return mat
