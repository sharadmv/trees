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

    @staticmethod
    def construct(points, constraints, X):
        if len(points) == 1:
            return InteractiveLeaf.construct(points, constraints, X)
        return InteractiveNonTerminal.construct(points, constraints, X)

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

    def reconfigure(self, df, constraints, new_constraint):
        children_points = [c.points() for c in self.children]
        left_points, right_points = children_points[0], children_points[1]
        points = left_points | right_points

        a, b, c = new_constraint
        if (a in left_points and b in left_points) or (a in right_points and b in right_points):
            return

        filtered_constraints = set()
        for constraint in constraints:
            (a2, b2, c2) = constraint
            if a2 not in points and b2 not in points and c2 not in points:
                continue
            filtered_constraints.add(constraint)

        graph = self.construct_constraint_graph(points, filtered_constraints)

        if a in left_points and c in left_points:
            incorrect = a
            move_into = 1
            points = left_points
        elif a in right_points and c in right_points:
            incorrect = a
            move_into = 0
            points = right_points
        elif b in left_points and c in left_points:
            incorrect = b
            move_into = 1
            points = left_points
        elif b in right_points and c in right_points:
            incorrect = b
            move_into = 0
            points = right_points

        move_points = set(n for n in nx.node_connected_component(graph, incorrect) if n in points)
        print "Move points", move_points
        nodes_to_move = [self.point_index(n) for n in move_points]
        nodes_to_move.sort(key=lambda x: len(x[1][0]))

        #if len(nodes_to_move) == len(children_points[1 - move_into]):
            #into_node = self.children[move_into]
            #move_into = None
        #else:
            #into_node = self

        for node, (_, _) in nodes_to_move:
            if self.children[1 - move_into].point_count() == 1:
                into_node = self.children[move_into]
            else:
                into_node = self
            parent = node.detach_node()
            assignment, _= into_node.sample_assignment(df,
                                                  filtered_constraints,
                                                  parent.points(),
                                                  force_side=move_into)
            into_node.attach_node(parent, assignment)

    def copy(self):
        node = InteractiveNonTerminal(self.left.copy(), self.right.copy(), None, self.state, self.time)
        node.left.parent = node
        node.right.parent = node
        return node

    def get_required(self, constraints, points):

        constraints = constraints.copy()
        requirements = set()

        children_points = [c.points() for c in self.children]

        if len(children_points) == 1:
            return None, constraints

        my_points = children_points[0] | children_points[1]

        for constraint in constraints.copy():
            a, b, c = constraint

            if (a in my_points and b not in my_points) or (b in my_points and a not in my_points):
                continue

            if a in points and b in points:
                constraints.remove(constraint)
                continue

            if a not in points and b not in points and c not in points:
                constraints.remove(constraint)
                continue

            if a in points:
                for i, child_points in enumerate(children_points):
                    if b in child_points:
                        requirements.add(i)
                        if c not in child_points:
                            constraints.remove(constraint)
                            break
            elif b in points:
                for i, child_points in enumerate(children_points):
                    if a in child_points:
                        requirements.add(i)
                        if c not in child_points:
                            constraints.remove(constraint)
                            break

        assert len(requirements) <= 1
        if len(requirements) != 1:
            return None, constraints
        return list(requirements)[0], constraints

    def get_banned(self, constraints, points):

        banned = set()

        constraints = constraints.copy()

        for constraint in constraints.copy():
            a, b, c = constraint

            for i, child in enumerate(self.children):
                if isinstance(child, InteractiveLeaf):
                    continue
                children_points = [ch.points() for ch in child.children]

                if c in points:
                    for j in xrange(2):
                        if a in children_points[j] and b in children_points[1 - j]:
                            banned.add(i)
                            constraints.remove(constraint)
        return banned, constraints

    def sample_assignment(self, df, constraints, points, index=(), force_side=None):

        logging.debug("Sampling assignment at index: %s, %s" % (str(index), str(constraints)))

        required, constraints = self.get_required(constraints, points)
        banned, constraints = self.get_banned(constraints, points)

        logging.debug("Required to go into child: %s" % str(required))
        logging.debug("Banned from going into child: %s" % str(banned))

        counts = [c.point_count() for c in self.children]
        total = float(sum(counts))
        left_prob = counts[0] / total
        u = np.random.random()
        if required is not None:
            idx = required
            choice = self.children[required]
            prob = 0
        else:
            if force_side is not None:
                idx = force_side
                choice = self.children[idx]
            elif u < left_prob:
                choice = self.left
                idx = 0
            else:
                choice = self.right
                idx = 1
            prob = np.log(counts[idx]) - np.log(total)

        no_diverge_prob = (df.cumulative_divergence(self.time) - df.cumulative_divergence(choice.time)) / \
            counts[idx]
        u = np.random.random()
        if required is not None:
            assignment, p = choice.sample_assignment(df, constraints, points, index=index + (idx,))
            return assignment, prob + p
        if idx in banned:
            sampled_time, _ = df.sample(self.time, choice.time, counts[idx])
            diverge_prob = df.log_pdf(sampled_time, self.time, counts[idx])
            prob += diverge_prob
            return (index + (idx,), sampled_time), prob
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
