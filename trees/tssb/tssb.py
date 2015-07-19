import logging
import numpy as np
import scipy.stats as stats

from ..distribution import Distribution

depth_weight = lambda a, l: lambda j: l ** j * a

class TSSB(Distribution):

    def __init__(self, parameter_process, *args, **kwargs):
        super(TSSB, self).__init__(*args, **kwargs)
        self.parameter_process = parameter_process
        self.root = None

    def generate_node(self, index):
        alpha, gamma = self.get_parameter("alpha"), self.get_parameter("gamma")
        node = Node(self, index, alpha(len(index)), gamma, self.parameter_process)
        return node

    def generate_root(self):
        if self.root is None:
            self.root = self.generate_node(())
        return self.root

    def get_node(self, index):
        assert self.root is not None, "No nodes exist"
        return self.root.get_node(index)

    def sample_one(self, return_updates=False):
        return self.uniform_index(np.random.random(), return_updates=return_updates)

    def add_point(self, i, index):
        self.generate_root().add_point(i, index)

    def remove_point(self, i):
        assert self.root is not None, "Root must exits"
        self.root.remove_point(i)
        if self.root.is_dead():
            self.root = None

    def uniform_index(self, u, return_updates=False):
        return self.find_node(u, self.root, return_updates=return_updates)

    def parameters(self):
        return {"alpha", "gamma"}

class Node(object):

    def __init__(self, tssb, index, alpha, gamma, parameter_process):
        self.tssb = tssb
        self.index = index
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
        self.parameter = self.parameter_process.generate()

    def get_node(self, index):
        if index == ():
            return self
        child, rest = index[0], index[1:]
        assert child in self.children
        return self.children[child].get_node(rest)

    def generate_child(self, c):
        if c not in self.children:
            self.children[c] = self.tssb.generate_node(self.index + (c,))
        return self.children[c]

    def remove_child(self, c):
        assert self.children[c].is_dead(), 'Cannot remove undead child'
        del self.children[c]

    def uniform_index(self, u):
        s = 0
        p = 1
        i = -1
        lower_edge = 0
        upper_edge = 0
        while u > s:
            lower_edge = upper_edge
            i += 1
            if i not in self.psi:
                self.psi[i] = stats.beta(1, self.gamma).rvs()
            s += p * self.psi[i]
            p *= (1 - self.psi[i])
            upper_edge = s
        return i, (u - lower_edge) / (upper_edge - lower_edge)

    def add_point(self, i, index):
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
            for child, child_node in self.children.items():
                child_node.remove_point(i)
                if child_node.is_dead():
                    self.remove_child(child)
            self.descendent_points.remove(i)
            self.path_count -= 1

    def is_dead(self):
        return self.path_count == 0 and self.point_count == 0

    def num_children(self):
        return len(self.children)

    def __repr__(self):
        return "Node<%f, %u, %u>" % (self.nu, self.point_count, self.path_count)
