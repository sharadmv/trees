import networkx as nx
import logging
import numpy as np
from node import Node, NonTerminal, Leaf

class InteractiveNode(Node):

    @staticmethod
    def construct_constraint_graph(points, constraints):
        graph = nx.Graph()
        for point in points:
            graph.add_node(point)
        for (a, b, _) in constraints:
            graph.add_edge(a, b)
        return graph

    def prune_constraints(self, constraints, points, idx):
        choice, other = self.children[idx], self.children[1 - idx]
        new_constraints = []
        choice_points, other_points = choice.points(), other.points()
        for constraint in constraints:
            a, b, c = constraint
            if a in points and b in choice_points and c in other_points:
                continue
            if b in points and a in choice_points and c in other_points:
                continue

            new_constraints.append(constraint)
        return new_constraints

    @staticmethod
    def construct(points, constraints, X):
        if len(points) == 1:
            return InteractiveLeaf.construct(points, constraints, X)
        return InteractiveNonTerminal.construct(points, constraints, X)

    def is_path_banned(self, constraints, points):
        my_points = self.points()

        for constraint in constraints:
            a, b, c = constraint

            if a in points and b not in my_points and c in my_points:
                return True
            if b in points and a not in my_points and c in my_points:
                return True
        return False

    def is_path_required(self, constraints, points):
        my_points = self.points()

        for point in points:
            for constraint in constraints:
                a, b, c = constraint

                if a == point:
                    if b in my_points:
                        return True
                elif b == point:
                    if a in my_points:
                        return True
        return False

    def is_required(self, constraints, points):
        my_points = self.points()
        for constraint in constraints:
            a, b, c = constraint
            if a in points and b in my_points and c in my_points:
                return True
            if b in points and a in my_points and c in my_points:
                return True
        return False

    def is_banned(self, constraints, points):
        for idx, child in enumerate(self.children):
            child_points = child.points()
            other_points = self.children[1 - idx].points()
            for constraint in constraints:
                a, b, c = constraint
                if c in points and ((b in child_points and a in other_points) or
                                    (a in child_points and b in other_points)):
                    return True
        return False

class InteractiveNonTerminal(NonTerminal, InteractiveNode):

    @staticmethod
    def construct(points, constraints, X):
        filtered_constraints = set()
        for constraint in constraints:
            a, b, c = constraint
            if a not in points or b not in points or c not in points:
                continue
            filtered_constraints.add(constraint)

        graph = InteractiveNode.construct_constraint_graph(points, filtered_constraints)
        components = sorted(list(nx.connected_components(graph)), key=lambda x: -len(x))
        left, right = set(), set()
        while len(components) > 0:
            comp = set(components.pop())
            if len(right) >= len(left):
                left |= comp
            else:
                right |= comp

        left_constraints, right_constraints = set(), set()
        for constraint in filtered_constraints:
            a, b, c = constraint
            if a in left and b in left and c in right:
                continue
            if a in right and b in right and c in left:
                continue
            if a in left and b in left and c in left:
                left_constraints.add(constraint)
            if a in right and b in right and c in right:
                right_constraints.add(constraint)
        left_node = InteractiveNode.construct(left, left_constraints, X)
        right_node = InteractiveNode.construct(right, right_constraints, X)

        node = InteractiveNonTerminal(left_node, right_node, None, (left_node.state + right_node.state) / 2.0,
                                      min(left_node.time, right_node.time) / 2.0)
        left_node.parent = node
        right_node.parent = node
        return node

    def copy(self):
        node = InteractiveNonTerminal(self.left.copy(), self.right.copy(), None, self.state, self.time)
        node.left.parent = node
        node.right.parent = node
        return node

    def sample_assignment(self, df, constraints, points, index=(), force_side=None):

        counts = [c.point_count() for c in self.children]
        total = sum(counts)

        for idx, child in enumerate(self.children):
            if child.is_required(constraints, points):
                constraints = self.prune_constraints(constraints, points, idx)
                assignment, p = self.children[idx].sample_assignment(df,
                                                                     constraints,
                                                                     points,
                                                                     index=index + (idx,))
                return assignment, p

        left_prob = counts[0] / total
        u = np.random.random()
        choice = None

        for i, child in enumerate(self.children):
            if child.is_path_required(constraints, points):
                idx = i
                choice = child
                break
            if child.is_path_banned(constraints, points):
                idx = 1 - i
                choice = self.children[idx]
                break

        if choice is None:
            if u < left_prob:
                choice = self.left
                idx = 0
            else:
                choice = self.right
                idx = 1

        prob = np.log(counts[idx]) - np.log(total)

        if choice.is_banned(constraints, points):
            sampled_time, _ = df.sample(self.time, choice.time, counts[idx])
            diverge_prob = df.log_pdf(sampled_time, self.time, counts[idx])
            prob += diverge_prob
            return (index + (idx,), sampled_time), prob

        constraints = self.prune_constraints(constraints, points, idx)

        no_diverge_prob = (df.cumulative_divergence(self.time) - df.cumulative_divergence(choice.time)) / \
            counts[idx]
        u = np.random.random()
        if u < np.exp(no_diverge_prob):
            prob += no_diverge_prob
            assignment, p = choice.sample_assignment(df, constraints, points, index=index + (idx,))
            return assignment, prob + p
        else:
            sampled_time, _ = df.sample(self.time, choice.time, counts[idx])
            diverge_prob = df.log_pdf(sampled_time, self.time, counts[idx])
            prob += diverge_prob
            return (index + (idx,), sampled_time), prob

class InteractiveLeaf(InteractiveNode, Leaf):

    @staticmethod
    def construct(points, constraints, X):
        assert len(points) == 1
        assert len(constraints) == 0
        point = list(points)[0]
        return InteractiveLeaf(None, point, X[point])

    def copy(self):
        return InteractiveLeaf(None, self.point, self.state)

    def is_required(self, constraints, points):
        return False

    def is_banned(self, constraints, points):
        return True
