import logging
import numpy as np
from .. import Tree

class DirichletDiffusionTree(Tree):

    def __init__(self, root=None, constraints=[], **params):
        super(DirichletDiffusionTree, self).__init__(root=root,
                                                     constraints=constraints,
                                                     **params)
        self._marg_log_likelihood = None

    def initialize_from_data(self, X):
        X = np.array(X)
        N, _ = X.shape
        points = set(xrange(N))
        super(DirichletDiffusionTree, self).initialize_assignments(points)
        self.reconfigure_subtree(self.root, X)

    def reconfigure_subtree(self, root, X):
        if root.is_root():
            root_time = 0.0
        else:
            root_time = root.get_state('time')
        for node in self.dfs(node=root):
            if node == root:
                node.set_state('time', root_time)
                node.set_state('latent_value', sum(n.get_state('latent_value') for n in node.children) /
                               float(len(node.children)))
            elif node.is_leaf():
                node.set_state('time', 1.0)
                node.set_state('latent_value', X[node.point].ravel())
            else:
                min_time = min(n.get_state('time') for n in node.children)
                new_time = min_time - (min_time - root_time) / 2.0
                node.set_state('time', new_time)
                node.set_state('latent_value', sum(n.get_state('latent_value') for n in node.children) /
                               float(len(node.children)))

    def get_assignment(self, node):
        return (node.get_index(), node.get_state('time'))

    def calculate_node_likelihood(self, node=None):
        node = node or self.root

        if node.is_leaf():
            return 1, 0, self.likelihood_model.transition_probability(node.parent, node)

        node_time = node.get_state('time')
        left_child, right_child = node.children
        path_count, tree_prob, data_prob = self.calculate_node_likelihood(node=left_child)

        if not node.is_root():
            tree_prob += self.df.log_no_divergence(node.parent.get_state('time'), node_time, path_count)
            tree_prob += self.df.log_divergence(node_time)
        data_prob += self.likelihood_model.transition_probability(node.parent, node)

        right_path_count, right_tree_prob, right_data_prob = self.calculate_node_likelihood(node=right_child)
        return path_count + right_path_count, tree_prob + right_tree_prob, data_prob + right_data_prob

    def calculate_marg_log_likelihood(self):
        assert self.root is not None
        _, tree_structure, data_structure = self.calculate_node_likelihood()
        self._marg_log_likelihood = tree_structure + data_structure

    def marg_log_likelihood(self):
        if self._marg_log_likelihood is None:
            self.calculate_marg_log_likelihood()
        return self._marg_log_likelihood

    def sample_assignment(self, node=None, constraints=None, points=None, index=None):
        node = node or self.root
        constraints = constraints or self.constraints
        points = points or set()
        index = index or ()
        df = self.df

        logging.debug("Sampling assignment at index: %s" % str(index))

        counts = [c.leaf_count() for c in node.children]
        logging.debug("Path counts: %s" % str(counts))
        total = float(sum(counts))

        for idx, child in enumerate(node.children):
            if child.is_required(constraints, points):
                constraints = node.prune_constraints(constraints, points, idx)
                logging.debug("Child is required: %u" % idx)
                return self.sample_assignment(node=node.children[idx],
                                                    constraints=constraints,
                                                    points=points,
                                                    index=index + (idx,))
        left_prob = counts[0] / total
        u = np.random.random()
        choice = None

        for i, child in enumerate(node.children):
            if child.is_path_required(constraints, points):
                idx = i
                choice = child
                break
            if child.is_path_banned(constraints, points):
                idx = 1 - i
                choice = node.children[idx]
                break

        if choice is None:
            if u < left_prob:
                choice = node.children[0]
                idx = 0
            else:
                choice = node.children[1]
                idx = 1

        prob = np.log(counts[idx]) - np.log(total)
        logging.debug("Branching: %f" % prob)

        node_time = node.get_state('time')
        choice_time = choice.get_state('time')

        if choice.is_banned(constraints, points):
            logging.debug("Child is banned")
            sampled_time, _ = df.sample(node_time, choice_time, counts[idx])
            diverge_prob = df.log_pdf(node_time, sampled_time, counts[idx])
            logging.debug("Diverging at %f: %f" % (sampled_time, diverge_prob))
            prob += diverge_prob
            return (index + (idx,), sampled_time), prob

        constraints = node.prune_constraints(constraints, points, idx)

        no_diverge_prob = (df.cumulative_divergence(node_time) - df.cumulative_divergence(choice_time)) / \
            counts[idx]
        u = np.random.random()
        if u < np.exp(no_diverge_prob):
            logging.debug("Not diverging: %f" % no_diverge_prob)
            prob += no_diverge_prob
            assignment, p = self.sample_assignment(node=node.children[idx],
                                                   constraints=constraints,
                                                   points=points,
                                                   index=index + (idx,))
            return assignment, prob + p
        else:
            sampled_time, _ = df.sample(node_time, choice_time, counts[idx])
            diverge_prob = df.log_pdf(sampled_time, node_time, counts[idx])
            logging.debug("Diverging at %f: %f" % (sampled_time, diverge_prob))
            prob += diverge_prob
            return (index + (idx,), sampled_time), prob

    def log_prob_assignment(self, assignment, node=None):
        node = node or self.root

        (idx, time) = assignment
        assert idx is not ()

        df = self.df

        first, rest = idx[0], idx[1:]

        counts = [c.leaf_count() for c in node.children]
        total = float(sum(counts))
        prob = np.log(counts[first]) - np.log(total)
        logging.debug("Branching prob: %f" % prob)

        node_time = node.get_state('time')

        if len(idx) == 1:
            diverge_prob = df.log_pdf(node_time, time, counts[first])
            logging.debug("Diverging at %f: %f" % (time, diverge_prob))
            return prob + diverge_prob

        choice = node.children[first]
        choice_time = choice.get_state('time')

        no_diverge_prob = (df.cumulative_divergence(node_time) - df.cumulative_divergence(choice_time)) / \
            counts[first]
        logging.debug("Not diverging: %f" % no_diverge_prob)

        return prob + no_diverge_prob + self.log_prob_assignment((rest, time), node=node.children[first])

    def assign_node(self, node, assignment):
        (idx, time) = assignment
        assignee = self.get_node(idx)
        assignee.attach(node)
        node.set_state('time', time)

    def sample_latent(self):
        for node in self.dfs():
            if not node.is_leaf():
                lv = self.likelihood_model.sample_transition(node, node.parent)
                node.set_state('latent_value', lv)

    def node_as_string(self, node):
        return str(node.get_state('time'))

    def get_parameters(self):
        return {
            "df",
            "likelihood_model",
        }
