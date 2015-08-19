import logging
import numpy as np

class Node(object):

    def __init__(self, parent, state, time):
        self.parent = parent
        self.state = state
        self.time = time

    def copy(self):
        raise NotImplementedError

    def points(self):
        raise NotImplementedError

    def nodes(self):
        raise NotImplementedError

    def point_count(self):
        raise NotImplementedError

    def point_index(self, point):
        raise NotImplementedError

    def uniform_index(self, u):
        raise NotImplementedError

    def detach_node(self):
        parent = self.parent
        assert parent is not None, "Cannot detach the root node"

        grandparent = self.parent.parent
        assert grandparent is not None, "Cannot detach a depth 1 node"

        sibling = parent.other_child(self)

        parent.remove_child(sibling)
        grandparent.replace_child(parent, sibling)
        parent.parent = None
        return parent, grandparent

class NonTerminal(Node):

    def __init__(self, left, right, parent, state, time):
        super(NonTerminal, self).__init__(parent, state, time)
        self.children = [left, right]

    @property
    def tree_size(self):
        return self.left.tree_size + self.right.tree_size + 1

    def sample_assignment(self, df, index=()):
        counts = [c.point_count() for c in self.children]
        total = float(sum(counts))
        left_prob = counts[0] / total
        u = np.random.random()
        if u < left_prob:
            choice = self.left
            idx = 0
        else:
            choice = self.right
            idx = 1
        prob = np.log(counts[idx]) - np.log(total)
        no_diverge_prob = (df.cumulative_divergence(self.time) - df.cumulative_divergence(choice.time)) / \
            counts[idx]
        u = np.random.random()
        if u < np.exp(no_diverge_prob):
            prob += no_diverge_prob
            assignment, p = choice.sample_assignment(df, index=index + (idx,))
            return assignment, prob + p
        else:
            sampled_time, _ = df.sample(self.time, choice.time, counts[idx])
            diverge_prob = df.log_pdf(sampled_time, self.time, counts[idx])
            prob += diverge_prob
            return (index + (idx,), sampled_time), prob

    def log_prob_assignment(self, df, assignment):
        index, time = assignment
        idx, rest = index[0], index[1:]
        choice = self.children[idx]
        counts = [c.point_count() for c in self.children]
        total = float(sum(counts))
        prob = np.log(counts[idx]) - np.log(total)
        if len(rest) == 0:
            diverge_prob = df.log_pdf(time, self.time, counts[idx])
            return prob + diverge_prob
        no_diverge_prob = (df.cumulative_divergence(self.time) - \
                           df.cumulative_divergence(choice.time)) / \
                           counts[idx]
        prob += no_diverge_prob
        child_prob = choice.log_prob_assignment(df, (rest, time))
        return prob + child_prob

    def update_latent(self, lm):
        if self.parent is not None:
            assert self.time > self.parent.time, "Bad times: %s, %s" % (self.time, self.parent.time)
        self.state = lm.sample_transition(self.time, self.parent, self.children)
        self.left.update_latent(lm)
        self.right.update_latent(lm)

    def point_index(self, point):
        if point in self.left.points():
            return self.left.point_index(point)
        if point in self.right.points():
            return self.right.point_index(point)

    def other_child(self, node):
        if node == self.left:
            return self.right
        if node == self.right:
            return self.left

    def remove_child(self, node):
        assert node in self.children
        self.children.remove(node)
        node.parent = None

    def replace_child(self, node1, node2):
        assert node1 in self.children
        assert node2 not in self.children
        if node1 == self.left:
            self.children[0] = node2
        if node1 == self.right:
            self.children[1] = node2
        else:
            pass
        node2.parent = self

    def attach_node(self, node, df, u=None):
        left_size, right_size = self.left.point_count(), self.right.point_count()
        total = float(left_size + right_size)
        parent_time = self.time
        if u is None:
            u = np.random.random()

        if u < left_size / total:
            replace = self.left
        else:
            replace = self.right
        end_time = min(replace.time, node.left.time)
        node.time = df.inverse_cumulative(np.random.random(), parent_time, end_time)
        self.replace_child(replace, node)
        node.children.append(replace)
        replace.parent = node
        return node


    def uniform_index(self, u, ignore_depth=0):
        if ignore_depth > 0:
            total = float(self.left.tree_size + self.right.tree_size)
            left_prob, right_prob = self.left.tree_size / total, self.right.tree_size / total
            if u < left_prob:
                return self.left.uniform_index(u / left_prob, ignore_depth=ignore_depth - 1)
            return self.left.uniform_index((u - left_prob) / right_prob, ignore_depth=ignore_depth - 1)
        stay_prob = 1.0 / self.tree_size
        if u < stay_prob:
            return self
        remainder = 1 - stay_prob
        total = float(self.left.tree_size + self.right.tree_size)
        left_prob, right_prob = self.left.tree_size / total * remainder, \
            self.right.tree_size / total * remainder
        if u < stay_prob + left_prob:
            return self.left.uniform_index((u - stay_prob) / left_prob, ignore_depth=0)
        return self.right.uniform_index((u - stay_prob - left_prob) / right_prob, ignore_depth=0)

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    def log_likelihood(self, df, likelihood_model, parent_time):
        left_path_count, left_prob, left_ll = self.left.log_likelihood(df, likelihood_model, parent_time)
        diverge_prob = (df.cumulative_divergence(parent_time) - df.cumulative_divergence(self.time)) / float(left_path_count) + \
            np.log(df.divergence(self.time)) + np.log(1.0 / left_path_count)
        right_path_count, right_prob, right_ll = self.right.log_likelihood(df, likelihood_model, self.time)

        if self.parent is not None:
            log_likelihood = likelihood_model.transition_probability(self.state, self.time,
                                                                     self.parent.state,
                                                                     self.parent.time)
        else:
            log_likelihood = likelihood_model.transition_probability(self.state, self.time,
                                                                     None,
                                                                     None)
        return left_path_count + right_path_count, left_prob + diverge_prob + right_prob, log_likelihood + left_ll + right_ll

    def copy(self):
        node = NonTerminal(self.left.copy(), self.right.copy(), None, self.state, self.time)
        node.left.parent = node
        node.right.parent = node
        return node

    def verify_times(self):
        assert self.time <= min(self.left.time, self.right.time), (self.time, min(self.left.time, self.right.time))
        if self.parent:
            assert self.time >= self.parent.time, self.time
        self.left.verify_times()
        self.right.verify_times()

    def nodes(self):
        return [self] + self.left.nodes() + self.right.nodes()

    def points(self):
        return self.left.points() | self.right.points()

    def point_count(self):
        return self.left.point_count() + self.right.point_count()

    def get_child(self, index):
        return self.children[index]

    def set_child(self, index, node):
        self.children[index] = node

    def index(self, node):
        return self.children.index(node)

    def __getitem__(self, key):
        if key == ():
            return self
        return self.children[key[0]][key[1:]]

class Leaf(Node):

    def __init__(self, parent, point, state):
        super(Leaf, self).__init__(parent, state, 1.0)
        self.point = point

    @property
    def tree_size(self):
        return 1

    def copy(self):
        return Leaf(None, self.point, self.state)

    def nodes(self):
        return [self]

    def point_index(self, point):
        assert point == self.point
        return self

    def log_likelihood(self, df, likelihood_model, parent_time):
        return 1, 0, likelihood_model.transition_probability(self.state, self.time, parent=self.parent.state, parent_time=self.parent.time)

    def verify_times(self):
        assert self.time == 1.0

    def update_latent(self, lm):
        return

    def points(self):
        return {self.point}

    def uniform_index(self, _, ignore_depth=0):
        return self

    def point_count(self):
        return 1
