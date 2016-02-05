import numpy as np
import logging

class SPRSampler(object):

    def __init__(self, tree, X):
        self.tree = tree
        self.X = X
        self.last_move = None
        self.likelihoods = []

    def initialize_assignments(self):
        self.tree.initialize_assignments(np.arange(self.X.shape[0]))

    def add_constraint(self, constraint):
        self.tree.add_constraint(constraint, self.X)

    def parent_move(self):
        logging.debug("Copying tree...")
        tree = self.tree.copy()

        old_likelihood = self.tree.marg_log_likelihood()
        logging.debug("Old Marginal Likelihood: %f" % old_likelihood)

        node = tree.choice()
        old_assignment = tree.get_assignment(node.parent)
        old_index, old_state = old_assignment
        subtree = node.detach()

        backward_likelihood = tree.log_prob_assignment(old_assignment)
        logging.debug("Backward Likelihood: %f" % backward_likelihood)

        points = set()
        if len(tree.constraints) > 0:
            points = subtree.points()

        (assignment, forward_likelihood) = tree.sample_assignment(constraints=tree.constraints,
                                                                  points=points,
                                                                  state=old_state)
        logging.debug("Candidate assignment: %s", str(assignment))
        (index, state) = assignment

        tree.assign_node(subtree, assignment)
        new_likelihood = tree.marg_log_likelihood()

        logging.debug("New Marginal Likelihood: %f" % old_likelihood)
        logging.debug("Forward Likelihood: %f" % forward_likelihood)

        a = min(1, np.exp(new_likelihood + backward_likelihood - old_likelihood - forward_likelihood))
        if np.random.random() < a:
            logging.debug("Accepted new tree with probability: %f" % a)
            self.tree = tree
            return
        logging.debug("Rejected new tree with probability: %f" % a)

    def sample(self):
        self.tree = self.tree.copy()
        self.parent_move()
        self.likelihoods.append(self.tree.marg_log_likelihood())
