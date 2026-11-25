import unittest

import numpy as np

from bonus_b import controller, smoke


class ControllerTests(unittest.TestCase):
    def test_calibration(self):
        np.testing.assert_allclose(smoke([.1, .3, .5, .7, .9]), [.4, .8, .2, .9, .7])

    def test_scenarios(self):
        cases = (
            (.1, (.8, .4, 0), 1307 / 41, 20),
            (.9, (0, .7, 1), 48575 / 739, 80),
            (.3, (.4, .8, 0), 1563 / 41, 50),
        )
        for x, expected_strengths, expected_centroid, expected_maximum in cases:
            with self.subTest(x=x):
                strengths, _, _, centroid, maximum = controller(x)
                np.testing.assert_allclose(strengths, expected_strengths)
                self.assertAlmostEqual(centroid, expected_centroid, places=5)
                self.assertEqual(maximum, expected_maximum)

    def test_ties_select_a_maximum(self):
        for x, expected in ((.2, 50), (57 / 70, 80)):
            with self.subTest(x=x):
                _, y, z, _, maximum = controller(x)
                self.assertEqual(maximum, expected)
                self.assertAlmostEqual(z[y == maximum].item(), z.max())
        for x, expected in ((.2 - 1e-8, 20), (.2 + 1e-8, 50),
                            (57 / 70 - 1e-8, 50), (57 / 70 + 1e-8, 80)):
            with self.subTest(x=x):
                self.assertEqual(controller(x)[-1], expected)

    def test_domain_endpoints(self):
        for x in (0, 1):
            _, _, _, centroid, maximum = controller(x)
            self.assertTrue(0 <= centroid <= 100)
            self.assertTrue(0 <= maximum <= 100)
        for x in (-.1, 1.1):
            with self.assertRaises(ValueError):
                controller(x)
