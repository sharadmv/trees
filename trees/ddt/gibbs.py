import random
import numpy as np

class GibbsSampler(object):

    def __init__(self, ddt):
        self.ddt = ddt

    def sample_node_parent(self):
        ddt = self.ddt.copy()
        node = ddt.choice()
        self.parent_move(ddt, node)

    def parent_move(self, ddt, node):

        removed, parent, old_transition_likelihood = ddt.remove_parent(node)
        transition_likelihood = ddt.attach_node(removed, parent)

        old_likelihood, likelihood = self.ddt.log_likelihood(), ddt.log_likelihood()
        a = np.exp(min(0, likelihood + old_transition_likelihood - old_likelihood - transition_likelihood))

        if random.random() > 1 - a:
            self.ddt = ddt
        return self.ddt

    def update_latent(self):
        self.ddt.update_latent()

    def gibbs_sample(self):
        self.sample_node_parent()
        self.ddt.root.verify_times()
        self.update_latent()
