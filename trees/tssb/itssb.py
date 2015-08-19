import numpy as np
import logging
from tssb import TSSB, Node

class InteractiveTSSB(TSSB):

    def __init__(self, depth_function, parameter_process, max_depth=20, constraints=[], *args, **kwargs):
        super(InteractiveTSSB, self).__init__(depth_function, parameter_process, max_depth=max_depth, *args, **kwargs)
        self.constraints = {}
        for constraint in constraints:
            self.add_constraint(constraint)

    def get_constraints(self, point):
        return self.constraints[point]

    def add_constraint(self, constraint):
        a, b, c = constraint
        if a not in self.constraints:
            self.constraints[a] = []
        if b not in self.constraints:
            self.constraints[b] = []
        if c not in self.constraints:
            self.constraints[c] = []
        if (b, c, True) not in self.constraints[a]:
            self.constraints[a].append((b, c, True))
        if (a, c, True) not in self.constraints[b]:
            self.constraints[b].append((a, c, True))
        if (a, b, False) not in self.constraints[c]:
            self.constraints[c].append((a, b, False))

    def sample_one(self, point=None):
        return self.uniform_index(np.random.random(), point=point, use_constraints=True)

    def uniform_index(self, u, point=None, use_constraints=True):
        return self.find_node(u, point=point, use_constraints=use_constraints)


    def find_node(self, u, point=None, use_constraints=True):
        assert point is not None, "Must supply point for sampling in itssb"
        root = self.generate_root()
        constraints = self.constraints.get(point, [])
        if not use_constraints:
            constraints = []
        logging.debug("Banned root: %s"% str(root.is_banned(constraints)))
        sub_points = self.root.sub_points()
        new_constraints = []
        for constraint in constraints:
            a, b, pos = constraint
            if not pos:
                if a not in sub_points or b not in sub_points:
                    continue
            new_constraints.append(constraint)
        constraints = new_constraints
        banned, constraints = root.is_banned(constraints)
        required, constraints = root.is_required(constraints)
        if banned and not required:
            logging.info("Point %u is banned from tree." % point)
            return None, None
        return root.find_node(u, (), constraints=constraints, max_depth=self.max_depth)

    def generate_node(self, depth, parent):
        gamma = self.get_parameter("gamma")
        node = InteractiveNode(self, parent, depth, self.depth_function.alpha(depth), gamma, self.parameter_process)
        return node


    def get_state(self):
        return {
            'parameter_process': self.parameter_process,
            'max_depth': self.max_depth,
            'constraints': self.constraints,
            'root': self.root.get_state()
        }

    @staticmethod
    def load(state, parameters):
        tssb = InteractiveTSSB(state['parameter_process'], parameters=parameters, max_depth=state['max_depth'])
        tssb.root = InteractiveNode.load(tssb, None, state['root'])
        tssb.constraints = state['constraints']
        return tssb

class InteractiveNode(Node):

    def __init__(self, tssb, parent, depth, alpha, gamma, parameter_process):
        super(InteractiveNode, self).__init__(tssb, parent, depth, alpha, gamma, parameter_process)

    def find_node(self, u, index, constraints=[], max_depth=20):
        logging.debug("Trying node %s" % str(index))
        required_children = set()
        for child_index, child_node in self.children.items():
            required , constraints = child_node.is_required(constraints)
            if required:
                required_children.add(child_index)
        if len(required_children) == 0 and (u < self.nu or len(index) == max_depth):
            return self, index
        u = (u - self.nu) / (1 - self.nu)
        c, u, new_constraints = self.uniform_index(u, constraints=constraints, required_children=required_children)
        return self.children[c].find_node(u, index + (c,), constraints=new_constraints, max_depth=max_depth)

    @staticmethod
    def load(tssb, parent, state):
        node = InteractiveNode(tssb, parent, state['depth'], state['alpha'], state['gamma'], tssb.parameter_process)

        node.points = state['points']
        node.descendent_points = state['descendent_points']

        node.path_count = state['path_count']
        node.point_count = state['point_count']
        node.psi = state['psi']
        node.max_child = state['max_child']
        node.children = {i: InteractiveNode.load(tssb, node, v) for i, v in state['children'].items()}
        return node

    def uniform_index(self, u, constraints=[], required_children=set()):
        s = 0
        p = 1
        i = -1
        lower_edge = 0
        upper_edge = 0
        banned_children = set()
        for child_index, child_node in self.children.items():
            banned, constraints = child_node.is_banned(constraints)
            if banned:
                banned_children.add(child_index)

        if len(required_children) > 0:
            logging.debug("Picking required child: %s, %s" % (list(required_children), constraints))
            return list(required_children)[0], u, constraints

        while u > s:
            lower_edge = upper_edge
            i += 1
            child = self.generate_child(i)
            required, constraints = child.is_required(constraints)
            psi = self.psi[i]
            if i in banned_children:
                logging.debug("Skipping banned child %u" % i)
                psi = 0
            s += p * psi
            p *= (1 - psi)
            upper_edge = s
            if required:
                logging.debug("Choosing required child %u" % i)
                return i, u * (upper_edge - lower_edge), constraints
        return i, (u - lower_edge) / (upper_edge - lower_edge), constraints

    def is_banned(self, constraints):
        sub_points = self.sub_points()
        new_constraints = []
        banned = False
        for constraint in constraints:
            a, b, pos = constraint
            if not pos:
                if a in self.points and b in sub_points:
                    logging.debug("Banned because %u in self and %u in sub" % (a, b))
                    logging.debug(self.points)
                    banned = banned | True
                    continue
                if b in self.points and a in sub_points:
                    logging.debug("Banned because %u in self and %u in sub" % (b, a))
                    logging.debug(self.points)
                    banned = banned | True
                    continue
                child_subpoints = {i: c.sub_points() for i, c in self.children.iteritems()}
                a_child = {i for i in self.children if a in child_subpoints[i]}
                b_child = {i for i in self.children if b in child_subpoints[i]}
                if a_child == b_child:
                    new_constraints.append(constraint)
                    banned = banned | False
                else:
                    logging.debug("Banned otherwise: %s" % (str((a_child, b_child))))
                    banned = banned | True
            else:
                new_constraints.append(constraint)
        return banned, new_constraints

    def is_required(self, constraints):
        sub_points = self.sub_points()
        new_constraints = []
        required = False
        for constraint in constraints:
            a, b, pos = constraint
            if pos:
                if a not in sub_points:
                    new_constraints.append(constraint)
                    continue
                if a in sub_points and b in sub_points:
                    required = required | True
                    new_constraints.append(constraint)
                    continue
                child_subpoints = {i: c.sub_points() for i, c in self.children.iteritems()}
                b_child = {i for i in self.children if b in child_subpoints[i]}
                if len(b_child) == 0:
                    required = required | True
                else:
                    required = required | False
            else:
                new_constraints.append(constraint)
        return required, new_constraints

