import logging
import numpy as np
import scipy.stats as stats

from ..distribution import Distribution

depth_weight = lambda a, l: lambda j: l ** j * a

class TSSB(Distribution):

    def __init__(self, parameter_process, *args, **kwargs):
        super(TSSB, self).__init__(*args, **kwargs)
        self.nodes = {}
        self.points = {}
        self.parameter_process = parameter_process

    def generate_node(self, index):
        alpha, gamma = self.get_parameter("alpha"), self.get_parameter("gamma")
        node = Node(alpha(len(index)), gamma, self.parameter_process)
        return node

    def sample_one(self, return_updates=False):
        return self.uniform_index(np.random.random(), return_updates=return_updates)

    def uniform_index(self, u, return_updates=False):
        return self.find_node(u, (), return_updates=return_updates)

    def find_node(self, u, index, return_updates=False):
        if index in self.nodes:
            node = self.nodes[index].copy()
        else:
            node = self.generate_node(index)
        node_updates = {index: node}
        if len(index) > 5:
            if return_updates:
                return index, node_updates
            return index
        if u < node.nu:
            if return_updates:
                return index, node_updates
            else:
                return index
        u = (u - node.nu) / (1 - node.nu)
        child, u = node.uniform_index(u)
        if return_updates:
            index, updates = self.find_node(u, index + (child,), return_updates=return_updates)
            node_updates.update(updates)
        else:
            index = self.find_node(u, index + (child,), return_updates)
        if return_updates:
            return index, node_updates
        else:
            return index

    def apply_node_updates(self, updates):
        self.nodes.update(updates)

    def get_node(self, index):
        if index not in self.nodes:
            self.nodes[index] = self.generate_node(index)
        return self.nodes[index]

    def add_point(self, i, index):
        assert i not in self.points
        logging.debug("Adding point %u to %s" % (i, str(index)))
        for n, next in self.path_iterator(index):
            node = self.get_node(n)
            node.add_descendent(i, next[-1])
        final_node = self.get_node(index)
        final_node.add_point(i)
        self.points[i] = index

    def size_biased_permutation(self, index):
        assert index in self.nodes
        node = self.get_node(index)
        if len(node.psi) <= 1:
            return

        labels, weights = zip(*node.psi.items())

        weights = np.array(weights)

        permutation = []
        while len(permutation) < len(labels):
            o = np.random.choice(labels, p=weights/np.sum(weights))
            weights[o] = 0
            permutation.append(o)

        nodes = {}
        psi, children = {}, {}
        for i, o in enumerate(permutation):

            to = index + (i,)
            fro = index + (o,)

            if to in self.nodes:
                for point in self.nodes[to].points:
                    logging.info("Updating %u from %s to %s" % (point, to, fro))
                    self.points[point] = fro
                    logging.info("Updated %u to %s" % (point, str(self.points[point])))


            if i in node.psi:
                psi[o] = node.psi[i]

            if i in node.children:
                children[o] = node.children[i]

            for n in self.nodes:
                if len(n) < len(index):
                    nodes[n] = self.nodes[n]
                    continue
                left, right = n[:len(index)], n[len(index):]
                if left != index:
                    nodes[n] = self.nodes[n]
                    continue
                if len(n) < len(to):
                    nodes[n] = self.nodes[n]
                    continue
                left, right = n[:len(to)], n[len(to):]
                if left == to:
                    nodes[fro + right] = self.nodes[n]

        node.psi = psi
        node.children = children
        self.nodes = nodes
        print self.points


    def remove_point(self, i):
        assert i in self.points, "%u not in points" % i
        index = self.points[i]
        logging.debug("Removing point %u from %s" % (i, str(index)))
        for n, next in self.path_iterator(index):
            node = self.get_node(n)
            node.remove_descendent(i, next[-1])
            if node.path_count == 0 and node.point_count == 0:
                del self.nodes[n]
        final_node = self.get_node(index)
        final_node.remove_point(i)
        if final_node.path_count == 0 and final_node.point_count == 0:
            del self.nodes[index]
        del self.points[i]

    def path_iterator(self, index):
        node = ()
        for i in index:
            yield node, node + (i,)
            node += (i, )

    def parameters(self):
        return {"alpha", "gamma"}

    def __getitem__(self, key):
        return self.nodes[key]

class Node(object):

    def __init__(self, alpha, gamma, parameter_process):
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
        self.parameter = self.parameter_process.generate()

    def add_descendent(self, i, index):
        for ind in xrange(index + 1):
            if ind not in self.psi:
                self.psi[ind] = stats.beta(1, self.gamma).rvs()
        if index not in self.children:
            self.children[index] = set()
        self.children[index].add(i)
        self.max_child = max(self.psi.keys())
        self.path_count += 1

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

    def copy(self):
        node = Node(self.alpha, self.gamma, self.parameter_process)
        node.path_count = self.path_count
        node.point_count = self.point_count
        node.nu = self.nu
        node.max_child = self.max_child
        node.points = self.points.copy()
        node.psi = self.psi.copy()
        node.children = self.children.copy()
        return node

    def remove_descendent(self, i, index):
        assert index in self.psi
        assert index in self.children
        self.children[index].remove(i)
        self.path_count -= 1
        if index == self.max_child and len(self.children[index]) == 0:
            del self.psi[index]
            if self.path_count > 0:
                self.max_child = max(self.psi.keys())
                for ind in self.psi:
                    if ind > self.max_child:
                        del self.psi[index]
        if not self.children[index]:
            del self.children[index]

    def add_point(self, i):
        self.points.add(i)
        self.point_count += 1

    def remove_point(self, i):
        assert i in self.points, "%u not in node's points" % i

        self.points.remove(i)
        self.point_count -= 1

    def __repr__(self):
        return "Node<%f, %u, %u>" % (self.nu, self.point_count, self.path_count)
