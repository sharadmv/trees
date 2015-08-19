import numpy as np
from trees.ddt import DirichletDiffusionTree, Inverse, GaussianLikelihoodModel, MetropolisHastingsSampler
import unittest

class TestDDT(unittest.TestCase):

    def setUp(self):
        self.N = 15
        self.D = 2
        self.df = Inverse(c=0.5)
        self.lm = GaussianLikelihoodModel(mu0=np.zeros(self.D),
                                          sigma0=np.eye(self.D),
                                          sigma=np.eye(self.D))
        self.ddt = DirichletDiffusionTree(self.df, self.lm)
        self.sampler = MetropolisHastingsSampler(self.ddt, np.zeros((self.N, self.D)))
        self.sampler.initialize_assignments()

    def test_choice(self):
        stay_prob = 1.0 / self.ddt.root.tree_size

        self.assertEqual(self.ddt.uniform_index(stay_prob / 2, ignore_depth=0), self.ddt.root)

        left_size, right_size = self.ddt.root.left.tree_size, self.ddt.root.right.tree_size

        remainder = 1 - stay_prob
        left_prob, right_prob = left_size / float(left_size + right_size) * remainder, \
            right_size / float(left_size + right_size) * remainder

        self.assertEqual(stay_prob + left_prob + right_prob, 1)

        p = 1.0 / left_size / 2

        self.assertEqual(self.ddt.uniform_index(stay_prob + p * left_prob, ignore_depth=0), self.ddt.root.left)

        p = 1.0 / right_size / 2
        self.assertEqual(self.ddt.uniform_index(stay_prob + left_prob + p * right_prob, ignore_depth=0), self.ddt.root.right)

    def test_get_node(self):
        self.assertEqual(self.ddt[()], self.ddt.root)

    def test_point_index(self):
        def find_point(i):
            for node in self.ddt.dfs():
                if {i} == node.points():
                    return node
        for i in xrange(self.N):
            self.assertEqual(self.ddt.point_index(i), find_point(i))

    def test_detach_node(self):
        self.assertRaises(AssertionError, lambda: self.ddt.root.detach_node())
        self.assertRaises(AssertionError, lambda: self.ddt.root.left.detach_node())
        self.assertRaises(AssertionError, lambda: self.ddt.root.right.detach_node())
        for i in xrange(self.N):
            node = self.ddt.point_index(0)
            if not node.parent is self.ddt.root:
                break


        old_parent = node.parent
        sibling = node.parent.other_child(node)
        old_grandparent = old_parent.parent

        parent, grandparent = node.detach_node()

        self.assertEqual(old_parent, parent)
        self.assertEqual(node.parent, parent)
        self.assertEqual(node.parent, old_parent)
        self.assertEqual(node.parent, old_parent)
        self.assertEqual(node.parent.parent, None)
        self.assertEqual(len(node.parent.children), 1)
        self.assertTrue(sibling in old_grandparent.children)

    def test_sample_assignment(self):
        for _ in xrange(1000):
            assignment, log_prob = self.ddt.sample_assignment()
            self.assertAlmostEqual(log_prob, self.ddt.log_prob_assignment(assignment), places=5)
