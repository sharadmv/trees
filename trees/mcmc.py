import logging
import random
import numpy as np

class MetropolisHastingsSampler(object):

    def __init__(self, tree, X):
        self.tree = tree
        self.X = X
        self.last_move = None

    def initialize_assignments(self):
        self.tree.initialize_from_data(self.X)

    def parent_move(self):
        tree = self.tree.copy()

        old_likelihood = tree.marg_log_likelihood()

        node = tree.choice()
        assignment = tree.get_assignment(node)
        parent = node.detach()

        (assignment, )

    def sample_node_parent(self):
        ddt = self.ddt.copy()
        node, assignment = ddt.choice(ignore_depth=2)
        assert node is not ddt.root
        assert node.parent is not ddt.root
        index, time = assignment
        index = index[:-1]
        self.parent_move(ddt, node, (index, time))

    def parent_move(self, tree, node, assignment, force=False):
        old_tree_likelihood = self.tree.marg_log_likelihood()

        points = set()
        if len(tree.constraints) > 0:
            points = parent.points()
        backward_prob = ddt.log_prob_assignment(assignment)

        time = float('inf')
        move = None
        try_counter = 0
        while time > parent.left.time:
            (move, time), forward_prob = ddt.sample_assignment(points=points)
            try_counter += 1
            if try_counter > 500:
                return
        ddt.attach_node(parent, (move, time))
        new_tree_likelihood = ddt.marg_log_likelihood()
        #logging.debug("Move: %s" % str((move, time)))
        #logging.debug("Forward: %f" % forward_prob)
        #logging.debug("Backward: %f" % backward_prob)
        #logging.debug("Probs: (%f, %f)" % (old_tree_likelihood, new_tree_likelihood))
        a = min(1, np.exp(new_tree_likelihood + backward_prob - old_tree_likelihood - forward_prob))
        #logging.debug("Accept Probability: %f" % a)
        if force or np.random.random() < a:
            self.last_move = self.ddt, assignment, (move, time)
            self.ddt = ddt

    def update_latent(self):
        ddt = self.ddt.copy()
        ddt.update_latent()
        self.ddt = ddt

    def sample(self):
        if np.random.random() < 0.5:
            self.sample_node_parent()
        else:
            self.update_latent()
