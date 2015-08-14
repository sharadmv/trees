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


class NonTerminal(Node):

    def __init__(self, left, right, parent, state, time):
        super(NonTerminal, self).__init__(parent, state, time)
        self.children = [left, right]

    def update_latent(self, lm):
        if self.parent is not None:
            assert self.time > self.parent.time, "Bad times: %s, %s" % (self.time, self.parent.time)
        if self.parent is not None:
            self.state = lm.sample_transition(self.time, self.parent.state, self.parent.time,
                                              [(self.left.state, self.left.time),
                                               (self.right.state, self.right.time)])
        else:
            self.state = lm.sample_transition(self.time, None, None,
                                              [(self.left.state, self.left.time),
                                               (self.right.state, self.right.time)])
        self.left.update_latent(lm)
        self.right.update_latent(lm)

    def point_index(self, point):
        if point in self.left.points():
            return self.left.point_index(point)
        if point in self.right.points():
            return self.right.point_index(point)

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

    def point_count(self):
        return 1
