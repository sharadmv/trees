"""
This file defines the global tree interface.
"""
import networkx as nx
import numpy as np
from functools import reduce
from itertools import combinations
from tqdm import tqdm

class Tree(object):

    def __init__(self, root=None, constraints=[], **params):
        self.root = root
        self.constraints = constraints
        self.parameters = {}
        for param in self.get_parameters():
            assert param in params, "Missing parameter: %s" % param
            self.parameters[param] = params[param]

    def leaf(self, *args, **kwargs):
        return TreeLeaf(*args, **kwargs)

    def node(self, *args, **kwargs):
        return TreeNode(*args, **kwargs)

    def initialize_assignments(self, points):
        self.root = TreeNode.construct(points, self.constraints)

    def set_root(self, root):
        self.root = root

    def copy(self):
        tree = self.__class__(root=self.root.copy(), **self.parameters.copy())
        return tree

    def choice(self):
        choice = None
        i = 0
        for node in self.dfs():
            if node.parent is None or node.parent.parent is None:
                continue
            p = i / (i + 1.0)
            if np.random.random() > p:
                choice = node
            i += 1
        return choice

    def __getattr__(self, key):
        params = self.__getattribute__("parameters")
        if key in params:
            return params[key]
        return self.__getattribute__(key)

    def dfs(self, node=None):
        if node is None:
            node = self.root
        if not node.is_leaf():
            for child in node.children:
                for traversed in self.dfs(node=child):
                    yield traversed
        yield node

    def generate_constraints(self):
        points = self.root.points()
        combos = list(combinations(points, 3))
        constraints = set()
        for (a, b, c) in tqdm(combos):
            if self.verify_constraint((a, b, c)):
                constraints.add((a, b, c))
                continue
            if self.verify_constraint((b, c, a)):
                constraints.add((b, c, a))
                continue
            if self.verify_constraint((c, a, b)):
                constraints.add((c, a, b))
                continue
        return constraints

    def verify_constraint(self, constraint):
        (a, b, c) = constraint
        ai = self.point_index(a)
        bi = self.point_index(b)
        ci = self.point_index(c)

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

    def score_constraints(self, constraints):
        score = 0
        for constraint in constraints:
            score += self.verify_constraint(constraint)
        return score

    def marg_log_likelihood(self):
        raise NotImplementedError

    def sample_assignment(self):
        raise NotImplementedError

    def assign_node(self, node, assignment):
        raise NotImplementedError

    def get_node(self, index):
        return self.root.get_node(index)

    def point_index(self, i):
        return self.root.point_index(i)

    def uniform_index(self, u):
        raise NotImplementedError

    def get_parameters(self):
        return set()

    def mrca(self, a, b):
        a_idx = self.point_index(a)
        b_idx = self.point_index(b)
        idx = ()
        i = 0
        while True:
            if a_idx[i] == b_idx[i]:
                idx += (a_idx[i],)
                i += 1
            else:
                return self.get_node(idx)

class TreeNode(object):

    def __init__(self, state=None):
        self.state = state or {}
        self.children = []
        self.parent = None

    def set_state(self, key, value):
        self.state[key] = value

    def get_state(self, key):
        return self.state[key]

    @staticmethod
    def construct_constraint_graph(points, constraints):
        graph = nx.Graph()
        for point in points:
            graph.add_node(point)
        for (a, b, _) in constraints:
            graph.add_edge(a, b)
        return graph

    @staticmethod
    def construct(points, constraints):
        if len(points) == 1:
            return TreeLeaf.construct(points, constraints)
        filtered_constraints = set()
        for constraint in constraints:
            a, b, c = constraint
            if a not in points or b not in points or c not in points:
                continue
            filtered_constraints.add(constraint)

        graph = TreeNode.construct_constraint_graph(points, filtered_constraints)
        components = sorted(list(nx.connected_components(graph)), key=lambda x: -len(x))
        left, right = set(), set()
        while len(components) > 0:
            comp = set(components.pop())
            if len(right) >= len(left):
                left |= comp
            else:
                right |= comp

        left_constraints, right_constraints = set(), set()
        for constraint in filtered_constraints:
            a, b, c = constraint
            if a in left and b in left and c in right:
                continue
            if a in right and b in right and c in left:
                continue
            if a in left and b in left and c in left:
                left_constraints.add(constraint)
            if a in right and b in right and c in right:
                right_constraints.add(constraint)
        left_node = TreeNode.construct(left, left_constraints)
        right_node = TreeNode.construct(right, right_constraints)

        node = TreeNode()
        node.add_child(left_node)
        node.add_child(right_node)
        return node

    def prune_constraints(self, constraints, points, idx):
        choice_points = self.children[idx].points()
        other_points = reduce(lambda x, y: x | y,
                              (c.points() for i, c in enumerate(self.children) if i != idx)
                              )
        new_constraints = []
        for constraint in constraints:
            a, b, c = constraint
            if a in points and b in choice_points and c in other_points:
                continue
            if b in points and a in choice_points and c in other_points:
                continue
            new_constraints.append(constraint)

        return new_constraints

    def copy(self):
        node = TreeNode(
            state=self.state.copy(),
        )
        for child in self.children:
            child = child.copy()
            child.parent = node
            node.add_child(child)
        return node

    def get_node(self, index):
        if index == ():
            return self
        first, rest = index[0], index[1:]
        assert first < len(self.children)
        return self.children[first].get_node(rest)

    def point_index(self, point, index=()):
        for i, child in enumerate(self.children):
            if point in child.points():
                return child.point_index(point, index=index+(i,))

    def set_assignment(self, assignment):
        raise NotImplementedError

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return False

    def add_child(self, child):
        self.children.append(child)
        child.set_parent(self)

    def remove_child(self, child):
        assert child in self.children
        self.children.remove(child)
        child.set_parent(None)

    def set_parent(self, parent):
        self.parent = parent

    def node_count(self):
        return 1 + sum(c.node_count() for c in self.children)

    def leaf_count(self):
        return sum(c.leaf_count() for c in self.children)

    def points(self):
        return reduce(set.union, (c.points() for c in self.children), set())

    def nodes(self):
        return reduce(set.union, (c.nodes() for c in self.children), set()) | self

    def detach(self):
        assert self.parent is not None, "Cannot detach root node"
        assert self.parent.parent is not None, "Cannot detach depth 1 node"

        parent, grandparent = self.parent, self.parent.parent
        grandparent.remove_child(parent)

        if len(parent.children) > 2:
            raise NotImplementedError("Haven't implemented detach for n-ary trees.")

        for sibling in parent.children:
            if sibling is not self:
                parent.remove_child(sibling)
                grandparent.add_child(sibling)
        return parent

    def attach(self, node):
        assert self.parent is not None, "Cannot attach to root node"

        parent = self.parent

        if len(parent.children) > 2:
            raise NotImplementedError("Haven't implemented attach for n-ary trees.")

        parent.remove_child(self)
        parent.add_child(node)
        node.add_child(self)

    def state_as_string(self):
        return str(self.state)

    def is_path_banned(self, constraints, points):
        my_points = self.points()

        for constraint in constraints:
            a, b, c = constraint

            if a in points and b not in my_points and c in my_points:
                return True
            if b in points and a not in my_points and c in my_points:
                return True
        return False

    def get_index(self):
        if self.parent is None:
            return ()
        return self.parent.get_index() + (self.parent.children.index(self),)

    def is_path_required(self, constraints, points):
        my_points = self.points()

        for point in points:
            for constraint in constraints:
                a, b, c = constraint

                if a == point:
                    if b in my_points:
                        return True
                elif b == point:
                    if a in my_points:
                        return True
        return False

    def is_required(self, constraints, points):
        my_points = self.points()
        for constraint in constraints:
            a, b, c = constraint
            if a in points and b in my_points and c in my_points:
                return True
            if b in points and a in my_points and c in my_points:
                return True
        return False

    def is_banned(self, constraints, points):
        for idx, child in enumerate(self.children):
            child_points = child.points()
            other_points = reduce(lambda x, y: x | y,
                              (c.points() for i, c in enumerate(self.children) if i != idx)
                              )
            for constraint in constraints:
                a, b, c = constraint
                if c in points and ((b in child_points and a in other_points) or
                                    (a in child_points and b in other_points)):
                    return True
        return False

class TreeLeaf(TreeNode):
    def __init__(self, point, state=None):
        super(TreeLeaf, self).__init__(state=state)
        self.children = None
        self.point = point

    @staticmethod
    def construct(points, constraints):
        assert len(points) == 1
        assert len(constraints) == 0
        point = list(points)[0]
        return TreeLeaf(point, state={})

    def copy(self):
        return TreeLeaf(self.point, state=self.state.copy())

    def is_leaf(self):
        return True

    def point_index(self, i, index=()):
        assert i == self.point, (i, self.point)
        return index

    def points(self):
        return {self.point}

    def node_count(self):
        return 1

    def leaf_count(self):
        return 1

    def state_as_string(self):
        return str(self.point)

    def is_required(self, points, constraints):
        return False

    def is_banned(self, points, constraints):
        return True

class MCMCSampler(object):

    def __init__(self, tree, X):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
