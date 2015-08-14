import logging
import numpy as np
import scipy.stats as stats

from ..distribution import Distribution
from .. import Tree

class TSSB(Tree, Distribution):

    def __init__(self, depth_function, parameter_process, max_depth=20, *args, **kwargs):
        super(TSSB, self).__init__(*args, **kwargs)
        self.depth_function = depth_function
        self.parameter_process = parameter_process
        self.root = None
        self.max_depth = max_depth

    def generate_node(self, depth, parent):
        gamma = self.get_parameter("gamma")
        node = Node(self, parent, depth, self.depth_function.alpha(depth), gamma, self.parameter_process)
        return node

    def generate_root(self):
        if self.root is None:
            self.root = self.generate_node(0, None)
        return self.root

    def marg_log_likelihood(self, X):
        log_likelihood = 0
        for node in self.dfs():
            for point in node.points:
                log_likelihood += self.parameter_process.data_log_likelihood(X[point], node.parameter)
        return log_likelihood

    def copy(self):
        tssb = TSSB(self.parameter_process, max_depth=self.max_depth, parameters=self.parameters)
        if self.root is not None:
            tssb.root = self.root.copy(None)
        return tssb

    def get_node(self, index):
        assert self.root is not None, "No nodes exist"
        return self.root.get_node(index)

    def point_index(self, i):
        assert self.root is not None, "No nodes in tree"
        assert i in self.root.points or i in self.root.descendent_points, "Point isn't added"
        return self.root.point_index(i, ())

    def sample_one(self, point=None):
        return self.uniform_index(np.random.random(), point=point)

    def add_point(self, i, index):
        logging.debug("Adding %i to %s" % (i, str(index)))
        self.generate_root().add_point(i, index)

    def remove_point(self, i):
        assert self.root is not None, "Root must exist"
        logging.debug("Removing %i" % i)
        self.root.remove_point(i)
        if self.root.is_dead():
            self.root = None

    def uniform_index(self, u, point=None):
        return self.find_node(u, point=point)

    def find_node(self, u, point=None):
        root = self.generate_root()
        return root.find_node(u, (), max_depth=self.max_depth)

    def garbage_collect(self):
        if self.root is not None:
            if self.root.is_dead():
                self.root = None
            else:
                self.root.garbage_collect()

    def dfs(self):
        assert self.root is not None
        yield self.root
        s = set(self.root.children.values())
        while len(s) > 0:
            child = s.pop()
            yield child
            s.update(child.children.values())

    def __getitem__(self, index):
        return self.get_node(index)

    def get_state(self):
        return {
            'depth_function': self.depth_function,
            'parameter_process': self.parameter_process,
            'max_depth': self.max_depth,
            'root': self.root.get_state()
        }

    @staticmethod
    def load(state, parameters):
        tssb = TSSB(state['depth_function'], state['parameter_process'], parameters=parameters, max_depth=state['max_depth'])
        tssb.root = Node.load(tssb, None, state['root'])
        return tssb

    def get_parameters(self):
        return {"gamma"}

class Node(Distribution):

    def __init__(self, tssb, parent, depth, alpha, gamma, parameter_process):
        self.tssb = tssb
        self.parent = parent
        self.depth = depth
        self.alpha = alpha
        self.gamma = gamma
        self.parameter_process = parameter_process

        self.path_count = 0
        self.point_count = 0

        self.nu = stats.beta(1, self.alpha).rvs()
        self.psi = {}

        self.max_child = -1
        self.points = set()
        self.children = {}
        self.descendent_points = set()
        if parent is not None:
            self.parameter = self.parameter_process.generate(parameter=parent.parameter)
        else:
            self.parameter = self.parameter_process.generate()

    def copy(self, parent):
        node = Node(self.tssb, parent, self.depth, self.alpha, self.gamma, self.parameter_process)
        node.path_count = self.path_count
        node.point_count = self.point_count

        node.nu = self.nu
        node.psi = self.psi.copy()

        node.max_child = self.max_child
        node.points = self.points.copy()
        node.descendent_points = self.descendent_points.copy()

        node.parent = parent

        children = {}
        for child, child_node in self.children.items():
            children[child] = child_node.copy(self)
        node.children = children
        return node

    def get_node(self, index):
        if index == ():
            return self
        child, rest = index[0], index[1:]
        assert child in self.children
        return self.children[child].get_node(rest)

    def sample_one(self):
        return self.parameter_process.sample_one(self.parameter)

    def point_index(self, i, index):
        if i in self.points:
            return self, index
        for c, child_node in self.children.items():
            if i in child_node.points:
                return child_node, index + (c,)
            if i in child_node.descendent_points:
                return child_node.point_index(i, index + (c,))

    def generate_child(self, c):
        if c not in self.children:
            self.children[c] = self.tssb.generate_node(self.depth + 1, self)
        if c > self.max_child:
            for i in xrange(self.max_child + 1, c + 1):
                self.psi[i] = stats.beta(1, self.gamma).rvs()
            self.max_child = c
        return self.children[c]

    def remove_child(self, c):
        assert self.children[c].is_dead(), 'Cannot remove undead child'
        del self.children[c]
        if c == self.max_child:
            new_max_child = -1 if len(self.children) == 0 else max(self.children.keys())
            for i in self.psi.keys():
                if i > new_max_child:
                    del self.psi[i]
            self.max_child = new_max_child

    def find_node(self, u, index, max_depth=20):
        if u < self.nu or len(index) == max_depth:
            return self, index
        u = (u - self.nu) / (1 - self.nu)
        c, u = self.uniform_index(u)
        return self.children[c].find_node(u, index + (c,))

    def uniform_index(self, u):
        s = 0
        p = 1
        i = -1
        lower_edge = 0
        upper_edge = 0
        while u > s:
            lower_edge = upper_edge
            i += 1
            self.generate_child(i)
            s += p * self.psi[i]
            p *= (1 - self.psi[i])
            upper_edge = s
        return i, (u - lower_edge) / (upper_edge - lower_edge)

    def add_point(self, i, index):
        assert i not in self.points and i not in self.descendent_points, "%u already in tree" % i
        if index == ():
            assert i not in self.points, "%u already in node's points" % i
            self.points.add(i)
            self.point_count += 1
        else:
            assert i not in self.descendent_points, "%u already in node's descendent points" % i
            child, rest = index[0], index[1:]
            self.generate_child(child).add_point(i, rest)
            self.descendent_points.add(i)
            self.path_count += 1

    def remove_point(self, i):
        if i not in self.points and i not in self.descendent_points:
            return
        if i in self.points:
            self.points.remove(i)
            self.point_count -= 1
        else:
            assert i in self.descendent_points, "%u not in node's descendent points" % i
            for c, child_node in self.children.items():
                child_node.remove_point(i)
                if child_node.is_dead():
                    self.remove_child(c)
            self.descendent_points.remove(i)
            self.path_count -= 1

    def garbage_collect(self):
        for c, child_node in self.children.items():
            if child_node.is_dead():
                self.remove_child(c)
            else:
                child_node.garbage_collect()

    def size_biased_permutation(self):
        weights = []
        for i in sorted(self.psi.keys()):
            weights.append(self.psi[i])

        permutation = []
        idx = np.arange(len(weights))
        while len(permutation) < len(weights):
            o = np.random.choice(idx, p=weights / np.sum(weights))
            permutation.append(o)
            weights[o] = 0.0
        psi, children = {}, {}
        for i, o in enumerate(permutation):
            psi[o] = self.psi[i]
            if i in self.children:
                children[o] = self.children[i]
        self.children = children
        self.psi = psi

    def is_dead(self):
        return self.path_count == 0 and self.point_count == 0

    def num_children(self):
        return len(self.children)

    def get_state(self):
        return {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'nu': self.nu,
            'psi': self.psi,
            'points': self.points,
            'descendent_points': self.descendent_points,
            'point_count': self.point_count,
            'path_count': self.path_count,
            'max_child': self.max_child,
            'depth': self.depth,
            'children': {i: v.get_state() for i, v in self.children.items()}
        }

    def sub_points(self):
        return self.points | self.descendent_points

    @staticmethod
    def load(tssb, parent, state):
        node = Node(tssb, parent, state['depth'], state['alpha'], state['gamma'], tssb.parameter_process)

        node.points = state['points']
        node.descendent_points = state['descendent_points']

        node.path_count = state['path_count']
        node.point_count = state['point_count']
        node.psi = state['psi']
        node.max_child = state['max_child']
        node.children = {i: Node.load(tssb, node, v) for i, v in state['children'].items()}
        return node

    def __repr__(self):
        return "Node<%f, %u, %u>" % (self.nu, self.point_count, self.path_count)
