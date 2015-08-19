import numpy as np
from trees.tssb import TSSB, GaussianParameterProcess, QuadraticDepth
import unittest

class TestAddRemove(unittest.TestCase):

    def setUp(self):

        D = 2
        mu0 = np.zeros(D)
        sigma0 = np.eye(D) * 4.0
        sigma = np.eye(D) / 4.0
        self.process = GaussianParameterProcess(mu0, sigma0, sigma0, sigma)
        self.df = QuadraticDepth(a=2, l=1)
        self.tssb = TSSB(self.df, self.process, parameters={
            'gamma': 0.7
        })

    def get_tssb(self, seed=1):
        return TSSB(self.df, self.process, parameters={
            'gamma': 0.7
        }, seed=seed)

    def test_simple_add(self):
        tssb = self.get_tssb()
        tssb.add_point(0, ())

        self.assertEqual(tssb.root.point_count, 1)
        self.assertEqual(tssb.root.path_count, 0)
        self.assertEqual(tssb.root.points, {0})
        self.assertEqual(tssb.root.descendent_points, set())

    def test_simple_add2(self):
        tssb = self.get_tssb()
        tssb.add_point(0, ())

        self.assertEqual(tssb.root.point_count, 1)
        self.assertEqual(tssb.root.path_count, 0)
        self.assertEqual(tssb.root.points, {0})
        self.assertEqual(tssb.root.descendent_points, set())

        self.assertRaises(AssertionError, lambda: tssb.add_point(0, ()))

        tssb.add_point(1, ())

        self.assertEqual(tssb.root.point_count, 2)
        self.assertEqual(tssb.root.path_count, 0)
        self.assertEqual(tssb.root.points, {0, 1})
        self.assertEqual(tssb.root.descendent_points, set())

        tssb.add_point(2, (0, ))

        self.assertEqual(tssb.root.point_count, 2)
        self.assertEqual(tssb.root.path_count, 1)
        self.assertEqual(tssb.root.points, {0, 1})
        self.assertEqual(tssb.root.descendent_points, {2})

        node = tssb.get_node((0, ))

        self.assertEqual(node.point_count, 1)
        self.assertEqual(node.path_count, 0)
        self.assertEqual(node.points, {2})
        self.assertEqual(node.descendent_points, set())


    def test_simple_add_remove(self):
        tssb = self.get_tssb()
        tssb.add_point(0, ())

        self.assertEqual(tssb.root.point_count, 1)
        self.assertEqual(tssb.root.path_count, 0)
        self.assertEqual(tssb.root.points, {0})
        self.assertEqual(tssb.root.descendent_points, set())

        self.assertRaises(AssertionError, lambda: tssb.add_point(0, ()))

        tssb.remove_point(0)

        self.assertEquals(tssb.root, None)

    def test_simple_add_remove2(self):
        tssb = self.get_tssb()
        tssb.add_point(0, ())
        tssb.add_point(1, (0,))

        tssb.remove_point(1)

        self.assertEquals(tssb.root.num_children(), 0)

        tssb.remove_point(0)

        self.assertEquals(tssb.root, None)


    def test_add_remove(self):
        tssb = self.get_tssb()
        tssb.add_point(0, ())
        tssb.add_point(1, (0,))

        tssb.remove_point(0)

        self.assertNotEquals(tssb.root, None)
        self.assertEquals(tssb.root.point_count, 0)
        self.assertEquals(tssb.root.path_count, 1)

        tssb.remove_point(1)
        self.assertEquals(tssb.root, None)

    def test_add_psi(self):
        tssb = self.get_tssb()
        tssb.add_point(0, ())

        self.assertEquals(tssb.root.psi, {})

        tssb.add_point(1, (0, ))

        self.assertEquals(set(tssb.root.psi.keys()), {0})

        tssb.add_point(2, (1, ))

        self.assertEquals(set(tssb.root.psi.keys()), {0, 1})

        tssb.add_point(3, (9, ))

        self.assertEquals(set(tssb.root.psi.keys()), set(range(10)))


    def test_add_remove_psi(self):
        tssb = self.get_tssb()

        tssb.add_point(0, (0, ))
        tssb.add_point(1, (1, ))

        tssb.add_point(2, (9, ))
        self.assertEquals(set(tssb.root.psi.keys()), set(range(10)))

        tssb.remove_point(1)
        self.assertEquals(set(tssb.root.psi.keys()), set(range(10)))

        tssb.remove_point(2)
        self.assertEquals(set(tssb.root.psi.keys()), {0})

        tssb.remove_point(0)
        self.assertEquals(tssb.root, None)

if __name__ == "__main__":
    unittest.main()
