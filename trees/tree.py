"""
This file defines the global tree interface.
"""
import networkx as nx
import numpy as np
from functools import reduce
from itertools import combinations
from tqdm import tqdm

from util import tree_cache

class Tree(object):

    def __init__(self, root=None, constraints=set(), **params):
        self.root = root
        self.constraints = frozenset(constraints)
        self.parameters = {}
        for param in self.get_parameters():
            assert param in params, "Missing parameter: %s" % param
            self.parameters[param] = params[param]

    def leaf(self, *args, **kwargs):
        return TreeLeaf(*args, **kwargs)

    def node(self, *args, **kwargs):
        return TreeNode(*args, **kwargs)

    def get_assignment(self, node):
        return (node.get_index(), node.state)

    def initialize_assignments(self, points):
        self.root = TreeNode.construct(points, self.constraints)


    def add_constraint(self, constraint, X):
        constraints = set(self.constraints)
        constraints.add(constraint)
        self.constraints = frozenset(constraints)
        a, b, c = constraint
        an, bn, cn = map(lambda p: self.get_node(self.point_index(p)), (a, b, c))
        subtree_root = self.mrca(an, self.mrca(bn, cn))
        subtree_root.reconfigure(self.constraints)
        self.reconfigure_subtree(subtree_root, X)

    def reconfigure_subtree(self, subtree):
        pass

    def set_root(self, root): self.root = root

    def copy(self):
        tree = self.__class__(root=self.root.copy(),
                              constraints=self.constraints,
                              **self.parameters.copy())
        return tree

    def node_as_string(self, node):
        return str(node.state)

    def choice(self):
        """
        Uses reservoir sampling to pick a node from a tree uniformly at random,
        ignoring nodes that are depth 0 and 1.
        """
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
            (an, bn, cn) = map(lambda p: self.get_node(self.point_index(p)), (a, b, c))
            a_b = self.mrca(an, bn)
            a_c = self.mrca(an, cn)
            b_c = self.mrca(bn, cn)
            if a_b == a_c:
                constraints.add((b, c, a))
            elif b_c == a_b:
                constraints.add((a, c, b))
            elif a_c == b_c:
                constraints.add((a, b, c))
        return constraints

    def verify_constraint(self, constraint):
        (a, b, c) = constraint
        (a, b, c) = map(lambda p: self.get_node(self.point_index(p)), (a, b, c))
        mrca = self.mrca(a, b)
        return mrca != self.mrca(mrca, c)

    def remove_point(self, i):
        node = self.get_node(self.point_index(i))
        return node.detach()

    def induced_subtree(self, points):
        subtree = self.copy()
        all_points = subtree.root.points()
        for point in all_points:
            if point not in points:
                subtree.remove_point(point)
        return subtree

    def verify_constraints(self, constraints):
        return all(self.verify_constraint(c) for c in constraints)

    def score_constraints(self, constraints):
        return sum(self.verify_constraint(c) for c in constraints)

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
        node_path = set([a])
        while not a.is_root():
            a = a.parent
            node_path.add(a)
        while b not in node_path:
            b = b.parent
        return b

class TreeNode(object):

    def __init__(self, state=None):
        self.state = state or {}
        self.children = []
        self.parent = None
        self.cache = {}

    def set_state(self, key, value):
        self.state[key] = value

    def get_state(self, key):
        return self.state[key]

    def set_cache(self, key, value):
        self.cache[key] = value
        pass

    def get_cache(self, key):
        return self.cache[key]

    def reconfigure(self, constraints):
        points = self.points()
        new_node = TreeNode.construct(points, constraints)
        self.children = new_node.children[:]
        for child in self.children:
            child.set_parent(self)

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


    @tree_cache("prune_constraints")
    def prune_constraints(self, constraints, points, idx):
        choice_points = self.children[idx].points()
        other_points = self.points() - choice_points
        new_constraints = set(constraints)
        for constraint in constraints:
            a, b, c = constraint
            if not (a in points and b in choice_points and c in other_points):
                continue
            if not (b in points and a in choice_points and c in other_points):
                continue
            new_constraints.remove(constraint)

        return frozenset(new_constraints)

    def copy(self):
        node = TreeNode(
            state=self.state.copy(),
        )
        node.cache = self.cache.copy()
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

    def dirty(self):
        self.cache = {}

    def add_child(self, child):
        self.children.append(child)
        child.set_parent(self)

    def remove_child(self, child):
        assert child in self.children
        self.children.remove(child)
        child.set_parent(None)

    def set_parent(self, parent):
        self.parent = parent

    @tree_cache("node_count")
    def node_count(self):
        return 1 + sum(c.node_count() for c in self.children)

    @tree_cache("leaf_count")
    def leaf_count(self):
        return sum(c.leaf_count() for c in self.children)

    @tree_cache("points")
    def points(self):
        return reduce(frozenset.union, (c.points() for c in self.children), frozenset())

    @tree_cache("nodes")
    def nodes(self):
        return reduce(set.union, (c.nodes() for c in self.children), set()) | self

    def detach(self):
        """
        Removes a subtree rooted at the current node from the tree.
        Returns the root of the subtree (this node).
        """
        assert self.parent is not None, "Cannot detach root node"
        if self.parent.is_root():
            parent = self.parent
            parent.remove_child(self)
            assert len(parent.children) == 1
            sibling = parent.children[0]
            parent.remove_child(sibling)
            for child in sibling.children:
                parent.add_child(child)
            return
        parent, grandparent = self.parent, self.parent.parent
        grandparent.remove_child(parent)

        if len(parent.children) > 2:
            raise NotImplementedError("Haven't implemented detach for n-ary trees.")

        for sibling in parent.children:
            if sibling is not self:
                parent.remove_child(sibling)
                grandparent.add_child(sibling)
        grandparent.make_dirty()
        self.parent.remove_child(self)
        return self

    def attach(self, node):
        assert self.parent is not None, "Cannot attach to root node"

        parent = self.parent

        if len(parent.children) > 2:
            raise NotImplementedError("Haven't implemented attach for n-ary trees.")

        parent.remove_child(self)
        new_parent = TreeNode()
        new_parent.add_child(self)
        new_parent.add_child(node)
        parent.add_child(new_parent)
        new_parent.make_dirty()


    @tree_cache("is_path_banned")
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

    def make_dirty(self):
        self.dirty()
        if self.parent is not None:
            self.parent.make_dirty()


    @tree_cache("is_path_required")
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

    @tree_cache("is_required")
    def is_required(self, constraints, points):
        my_points = self.points()
        for constraint in constraints:
            a, b, c = constraint
            if a in points and b in my_points and c in my_points:
                return True
            if b in points and a in my_points and c in my_points:
                return True
        return False

    @tree_cache("is_banned")
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
        return frozenset({self.point})

    def node_count(self):
        return 1

    def leaf_count(self):
        return 1

    def is_required(self, points, constraints):
        return False

    def is_banned(self, points, constraints):
        return True

class MCMCSampler(object):

    def __init__(self, tree, X):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
