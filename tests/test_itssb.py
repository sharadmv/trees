import numpy as np
from trees.tssb import InteractiveTSSB, GaussianParameterProcess, depth_weight
import unittest

class TestAddRemove(unittest.TestCase):

    def setUp(self):

        D = 2
        mu0 = np.zeros(D)
        sigma0 = np.eye(D) * 4.0
        sigma = np.eye(D) / 4.0
        self.process = GaussianParameterProcess(mu0, sigma0, sigma0, sigma)

    def get_tssb(self, seed=1):
        return InteractiveTSSB(self.process, parameters={
            'alpha': depth_weight(2, 1),
            'gamma': 0.7
        }, seed=seed)

    def test_simple_ban(self):
        tssb = self.get_tssb()

        tssb.add_point(0, (0,))
        tssb.add_point(1, (1,))

        self.assertEqual(tssb.root.is_banned([(0, 1, False)]), (True, []))

        tssb.remove_point(0)
        tssb.remove_point(1)

        tssb.add_point(0, (0,))
        tssb.add_point(1, (0, 1))

        self.assertEqual(tssb.root.is_banned([(0, 1, False)]), (False, [(0, 1, False)]))
        self.assertEqual(tssb.get_node((0,)).is_banned([(0, 1, False)]), (True, []))
        self.assertEqual(tssb.get_node((0,)).is_banned([(0, 1, False), (4, 5, True)]), (True, [(4, 5, True)]))


    def test_simple_require(self):
        tssb = self.get_tssb()

        tssb.add_point(0, (0,))
        tssb.add_point(2, (1,))

        self.assertEqual(tssb.root.is_required([(0, 2, True)]), (True, [(0, 2, True)]))
        self.assertEqual(tssb.get_node((0,)).is_required([(0, 2, True)]), (True, []))

if __name__ == "__main__":
    unittest.main()
