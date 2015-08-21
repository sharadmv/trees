import numpy as np
from node import Node, NonTerminal, Leaf

class InteractiveNode(Node):
    pass

class InteractiveNonTerminal(NonTerminal, InteractiveNode):

    def sample_assignment(self, df, constraints, points, index=()):
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

class InteractiveLeaf(Node, Leaf):
    pass
