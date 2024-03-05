import unittest

import numpy as np

from causalgen import Generator
from causalgen.variables import Node
from test.test_utils import DISTRIBUTIONS, SIZE


class TestGeneration(unittest.TestCase):
    def test_distributions(self):
        for dist, (kw1, kw2) in DISTRIBUTIONS.items():
            dg, rand = Generator(seed=0), np.random.default_rng(0)
            # 1. retrieve operations and check consistency of generator result
            dg_dist, np_dist = getattr(dg, dist), rand.normal if dist == 'noise' else getattr(rand, dist)
            node = dg_dist(**kw1, hidden=False, name='node')
            vec = np_dist(**kw2, size=SIZE)
            self.assertIsInstance(node, Node, f"Dist '{dist}': generator should return a Node instance")
            # 2. generate the node value and check that the two vectors coincide
            val = dg.generate(SIZE)['node'].values
            self.assertListEqual(list(val), list(vec), f"Dist '{dist}': wrong value returned")

    def test_random(self):
        dg, rand = Generator(), np.random.default_rng(0)
        # test None seed on init and reset
        self.assertIs(Generator(seed=None).random, np.random, f"None seed should return np.random generator")
        self.assertIs(dg.reset_seed(seed=None).random, np.random, f"None seed should return np.random generator")
        # test rng seed on init and reset
        self.assertIs(Generator(seed=rand).random, rand, f"Rng seed should the given generator")
        self.assertIs(dg.reset_seed(seed=rand).random, rand, f"Rng seed should the given generator")
        # test int seed on init and reset
        for gen in [Generator(0), dg.reset_seed(0)]:
            gv = gen.random.random(size=SIZE)
            rv = np.random.default_rng(0).random(size=SIZE)
            self.assertListEqual(list(gv), list(rv), f"Int seed should return the same vectors")

    # noinspection PyPep8Naming
    def test_generation(self):
        dg, rand = Generator(seed=0), np.random.default_rng(0)
        # correct values
        h = rand.uniform(size=SIZE)
        z = rand.binomial(n=1, p=0.5, size=SIZE)
        noise_1 = rand.normal(size=SIZE)
        noise_2 = rand.normal(size=SIZE)
        x = (2 * z - 1) * h + 0.01 * noise_1
        y = (x ** 2) * (0.01 * noise_2 + 1)
        # generator nodes
        H = dg.uniform(hidden=True, name='h')
        Z = dg.binomial(hidden=False, name='z')
        X = dg.descendant((2 * Z - 1) * H + 0.01 * dg.noise(), name='x')
        dg.descendant((X ** 2) * (0.01 * dg.noise() + 1), name='y')
        # check that generated values are correct
        df = dg.generate(num=SIZE, hidden=True)
        self.assertTrue(np.allclose(df['h'], h), f"Wrong values for 'h'")
        self.assertTrue(np.allclose(df['z'], z), f"Wrong values for 'z'")
        self.assertTrue(np.allclose(df['noise_1'], noise_1), f"Wrong values for 'noise_1'")
        self.assertTrue(np.allclose(df['noise_2'], noise_2), f"Wrong values for 'noise_2'")
        self.assertTrue(np.allclose(df['x'], x), f"Wrong values for 'x'")
        self.assertTrue(np.allclose(df['y'], y), f"Wrong values for 'y'")
        # check that hidden=False returns visible nodes only
        df = dg.generate(num=SIZE, hidden=False)
        self.assertListEqual(['z', 'x', 'y'], list(df.columns), f"Wrong nodes returned by generation")
