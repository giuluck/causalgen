import unittest
from typing import Dict, Tuple

import numpy as np

from causalgen import Generator
from causalgen.variables import Intermediate, Node

size = 10

rng = np.random.default_rng(0)

unary_operations: Dict[str, np.ndarray] = {
    'invert': rng.integers(-100, 100, size=size),
    'negative': rng.uniform(-100, 100, size=size),
    'abs': rng.uniform(-100, 100, size=size),
    'floor': rng.uniform(-100, 100, size=size),
    'ceil': rng.uniform(-100, 100, size=size),
    'trunc': rng.uniform(-100, 100, size=size),
    'round': rng.uniform(-100, 100, size=size),
    'sin': rng.uniform(-100, 100, size=size),
    'cos': rng.uniform(-100, 100, size=size),
    'tan': rng.uniform(-100, 100, size=size),
    'arcsin': rng.uniform(-1, 1, size=size),
    'arccos': rng.uniform(-1, 1, size=size),
    'arctan': rng.uniform(-100, 100, size=size),
    'sinh': rng.uniform(-100, 100, size=size),
    'cosh': rng.uniform(-100, 100, size=size),
    'tanh': rng.uniform(-100, 100, size=size),
    'arcsinh': rng.uniform(-100, 100, size=size),
    'arccosh': rng.uniform(1, 100, size=size),
    'arctanh': rng.uniform(-1, 1, size=size),
    'exp': rng.uniform(-100, 100, size=size),
    'exp2': rng.uniform(-100, 100, size=size),
    'expm1': rng.uniform(-100, 100, size=size),
    'log': rng.uniform(0, 100, size=size),
    'log2': rng.uniform(0, 100, size=size),
    'log10': rng.uniform(0, 100, size=size),
    'log1p': rng.uniform(-1, 100, size=size),
    'square': rng.uniform(-100, 100, size=size),
    'cbrt': rng.uniform(-100, 100, size=size),
}

binary_operations: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    'add': (rng.uniform(-100, 100, size=size), rng.uniform(-100, 100, size=size)),
    'subtract': (rng.uniform(-100, 100, size=size), rng.uniform(-100, 100, size=size)),
    'multiply': (rng.uniform(-100, 100, size=size), rng.uniform(-100, 100, size=size)),
    'true_divide': (rng.uniform(1, 100, size=size), rng.uniform(1, 100, size=size)),
    'floor_divide': (rng.integers(1, 100, size=size), rng.integers(1, 100, size=size)),
    'power': (rng.uniform(1, 100, size=size), rng.uniform(1, 100, size=size)),
    'mod': (rng.integers(1, 100, size=size), rng.integers(1, 100, size=size)),
    'bitwise_and': (rng.integers(0, 2, size=size), rng.integers(0, 2, size=size)),
    'bitwise_or': (rng.integers(0, 2, size=size), rng.integers(0, 2, size=size)),
    'bitwise_xor': (rng.integers(0, 2, size=size), rng.integers(0, 2, size=size))
}

distributions: Dict[str, Tuple[dict, dict]] = {
    'noise': (dict(amount=10), dict(loc=0, scale=10)),
    'uniform': (dict(low=-10, high=10), dict(low=-10, high=10)),
    'normal': (dict(mu=5, sigma=10), dict(loc=5, scale=10)),
    'lognormal': (dict(mu=5, sigma=10), dict(mean=5, sigma=10)),
    'exponential': (dict(scale=10), dict(scale=10)),
    'poisson': (dict(lam=10), dict(lam=10)),
    'geometric': (dict(p=0.8), dict(p=0.8)),
    'binomial': (dict(p=0.8), dict(n=1, p=0.8)),
    'integers': (dict(low=-10, high=10, endpoint=False), dict(low=-10, high=10, endpoint=False)),
    'choice': (
        dict(categories=['a', 'b', 'c'], replace=True, p=[0.1, 0.2, 0.7]),
        dict(a=['a', 'b', 'c'], replace=True, p=[0.1, 0.2, 0.7])
    )
}

aliases: Dict[str, Tuple[str, dict]] = {
    'noise': ('noise', dict()),
    'uniform': ('uni', dict()),
    'normal': ('norm', dict()),
    'lognormal': ('lnorm', dict()),
    'exponential': ('exp', dict()),
    'poisson': ('pois', dict()),
    'geometric': ('geom', dict()),
    'binomial': ('bin', dict()),
    'integers': ('int', dict()),
    'choice': ('cat', dict(categories=['a', 'b', 'c'])),
    'custom': ('var', dict(distribution=lambda s: rng.random(size=s)))
}


class TestCase(unittest.TestCase):
    def test_intermediate_operations(self):
        # TEST UNARY OPERATIONS
        for op, vec in unary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            var = Intermediate(lambda: vec, inputs=[])
            self.assertIs(var.value, vec, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the unary operation and check that the resulting value *is not* a new object
            op_var = operation(var)
            self.assertIs(var, op_var, f"Op '{op}': unary ops is not operating inplace")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec)
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")
        # TEST BINARY OPERATIONS (CONSTANT RHS)
        for op, (vec1, vec2) in binary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            var = Intermediate(lambda: vec1, inputs=[])
            self.assertIs(var.value, vec1, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the operation against a (single) constant value and check accordingly
            op_var = operation(var, vec2[0])
            self.assertIs(var, op_var, f"Op '{op}': unary ops is not operating inplace")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec1, vec2[0])
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")
        # TEST BINARY OPERATIONS (CONSTANT LHS)
        for op, (vec1, vec2) in binary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            var = Intermediate(lambda: vec1, inputs=[])
            self.assertIs(var.value, vec1, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the operation against a (single) constant value and check accordingly
            op_var = operation(vec2[0], var)
            self.assertIs(var, op_var, f"Op '{op}': unary ops is not operating inplace")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec2[0], vec1)
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")
        # TEST BINARY OPERATIONS (BOTH VARIABLES)
        for op, (vec1, vec2) in binary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            var1, var2 = Intermediate(lambda: vec1, inputs=[]), Intermediate(lambda: vec2, inputs=[])
            self.assertIs(var1.value, vec1, f"Op '{op}': intermediate variable is returning a wrong value")
            self.assertIs(var2.value, vec2, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the operation and check that the resulting value *is* a new object
            op_var = operation(var1, var2)
            self.assertIsNot(var1, op_var, f"Op '{op}': binary ops should not operate inplace")
            self.assertIsNot(var2, op_var, f"Op '{op}': binary ops should not operate inplace")
            self.assertIsInstance(op_var, Intermediate, f"Op '{op}': binary ops should return a new Intermediate")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec1, vec2)
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")

    def test_node_operations(self):
        # TEST UNARY OPERATIONS
        for op, vec in unary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            node = Node(generator=None, func=lambda: vec, parents=set(), hidden=False, name=op)
            node.sample()
            self.assertIs(node.value, vec, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the unary operation and check that the resulting value is a new Intermediate object
            op_var = operation(node)
            self.assertIsInstance(op_var, Intermediate, f"Op '{op}': node ops should return Intermediate variable")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec)
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")
        # TEST BINARY OPERATIONS (CONSTANT RHS)
        for op, (vec1, vec2) in binary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            node = Node(generator=None, func=lambda: vec1, parents=set(), hidden=False, name=op)
            node.sample()
            self.assertIs(node.value, vec1, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the operation against a (single) constant value and check accordingly
            op_var = operation(node, vec2[0])
            self.assertIsInstance(op_var, Intermediate, f"Op '{op}': node ops should return Intermediate variable")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec1, vec2[0])
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")
        # TEST BINARY OPERATIONS (CONSTANT LHS)
        for op, (vec1, vec2) in binary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            node = Node(generator=None, func=lambda: vec1, parents=set(), hidden=False, name=op)
            node.sample()
            self.assertIs(node.value, vec1, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the operation against a (single) constant value and check accordingly
            op_var = operation(vec2[0], node)
            self.assertIsInstance(op_var, Intermediate, f"Op '{op}': node ops should return Intermediate variable")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec2[0], vec1)
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")
        # TEST BINARY OPERATIONS (BOTH VARIABLES)
        for op, (vec1, vec2) in binary_operations.items():
            # 1. check that the intermediate variable is returning the correct vector
            operation = getattr(np, op)
            node1 = Node(generator=None, func=lambda: vec1, parents=set(), hidden=False, name=op)
            node2 = Node(generator=None, func=lambda: vec2, parents=set(), hidden=False, name=op)
            node1.sample()
            node2.sample()
            self.assertIs(node1.value, vec1, f"Op '{op}': intermediate variable is returning a wrong value")
            self.assertIs(node2.value, vec2, f"Op '{op}': intermediate variable is returning a wrong value")
            # 2. perform the operation and check that the resulting value *is* a new object
            op_var = operation(node1, node2)
            self.assertIsInstance(op_var, Intermediate, f"Op '{op}': node ops should return Intermediate variable")
            # 3. check that the operation on the variable returns the same value as the operation on the vector
            op_vec = operation(vec1, vec2)
            self.assertListEqual(list(op_var.value), list(op_vec), f"Op '{op}': wrong value returned")

    def test_distributions(self):
        for dist, (kw1, kw2) in distributions.items():
            dg, rand = Generator(seed=0), np.random.default_rng(0)
            # 1. retrieve operations and check consistency of generator result
            dg_dist, np_dist = getattr(dg, dist), rand.normal if dist == 'noise' else getattr(rand, dist)
            node = dg_dist(**kw1, hidden=False, name='node')
            vec = np_dist(**kw2, size=size)
            self.assertIsInstance(node, Node, f"Dist '{dist}': generator should return a Node instance")
            # 2. generate the node value and check that the two vectors coincide
            val = dg.generate(size)['node'].values
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
            gv = gen.random.random(size=size)
            rv = np.random.default_rng(0).random(size=size)
            self.assertListEqual(list(gv), list(rv), f"Int seed should return the same vectors")

    def test_sources(self):
        dg = Generator()
        for operator, (alias, kwargs) in aliases.items():
            operation = getattr(dg, operator)
            # check that source node with given name is correctly built
            node = operation(**kwargs, name=operator)
            self.assertEqual(node.name, operator, f"Node was created with wrong name")
            self.assertIn(node, dg.nodes, f"Node was not inserted in generator structure")
            # check that node with same name raises an exception
            with self.assertRaises(AssertionError):
                operation(**kwargs, name=operator)
            # check that source nodes without names are correctly named
            for i in range(3):
                node = operation(**kwargs)
                self.assertEqual(node.name, f"{alias}_{i + 1}", f"Node was created with wrong name")
                self.assertIn(node, dg.nodes, f"Node was not inserted in generator structure")
        # check signature assertion in custom sources
        with self.assertRaises(AssertionError):
            dg.custom(lambda: np.empty(1))
        with self.assertRaises(AssertionError):
            dg.custom(lambda a, b: np.empty(1))

    # noinspection PyPep8Naming
    def test_descendants(self):
        dg, rand = Generator(seed=0), np.random.default_rng(0)
        # create sources
        A = dg.normal(hidden=False, name='a')
        vA = rand.normal(size=size)
        self.assertIs(A.hidden, False, f"Node A should be visible")
        self.assertIs(A.visible, True, f"Node A should be visible")
        B = dg.integers(hidden=True, name='b')
        vB = rand.integers(0, 1, endpoint=True, size=size)
        self.assertIs(B.hidden, True, f"Node B should be hidden")
        self.assertIs(B.visible, False, f"Node B should be hidden")
        Ext = Node(generator=None, func=lambda: np.empty(1), parents=set(), hidden=False, name='ext')
        # create correct descendants
        C = dg.descendant(lambda a, b: a + b, noise=None, hidden=False, name='c')
        self.assertIs(C.hidden, False, f"Node C should be visible")
        self.assertIs(C.visible, True, f"Node C should be visible")
        self.assertEqual(C.name, 'c', f"Node was created with wrong name")
        self.assertIn(C, dg.nodes, f"Node was not inserted in generator structure")
        D = dg.descendant(lambda x, y: x + y, noise=None, hidden=True, parents=[A, B], name='d')
        self.assertIs(D.hidden, True, f"Node D should be hidden")
        self.assertIs(D.visible, False, f"Node D should be hidden")
        self.assertEqual(D.name, 'd', f"Node was created with wrong name")
        self.assertIn(D, dg.nodes, f"Node was not inserted in generator structure")
        Var_1 = dg.descendant(lambda x, y: x + y, noise=0.1, hidden=False, parents=['a', 'b'])
        self.assertIs(Var_1.hidden, False, f"Node Var_1 should be visible")
        self.assertIs(Var_1.visible, True, f"Node Var_1 should be visible")
        self.assertEqual(Var_1.name, 'var_1', f"Node was created with wrong name")
        self.assertIn(Var_1, dg.nodes, f"Node was not inserted in generator structure")
        Var_2 = dg.descendant(A + B, noise=None, hidden=True)
        self.assertIs(Var_2.hidden, True, f"Node Var_2 should be hidden")
        self.assertIs(Var_2.visible, False, f"Node Var_2 should be hidden")
        self.assertEqual(Var_2.name, 'var_2', f"Node was created with wrong name")
        self.assertIn(Var_2, dg.nodes, f"Node was not inserted in generator structure")
        Var_3 = dg.descendant(A + B, noise=0.1, hidden=False)
        self.assertIs(Var_3.hidden, False, f"Node Var_3 should be visible")
        self.assertIs(Var_3.visible, True, f"Node Var_3 should be visible")
        self.assertEqual(Var_3.name, 'var_3', f"Node was created with wrong name")
        self.assertIn(Var_3, dg.nodes, f"Node was not inserted in generator structure")
        # check generated values (and noises)
        df = dg.generate(num=size, hidden=True)
        n1, n3 = rand.normal(scale=0.1, size=size), rand.normal(scale=0.1, size=size)
        self.assertListEqual(list(df['a']), list(vA), f"Wrong samples from 'a'")
        self.assertListEqual(list(df['b']), list(vB), f"Wrong samples from 'b'")
        self.assertListEqual(list(df['c']), list(vA + vB), f"Wrong operation from 'c'")
        self.assertListEqual(list(df['d']), list(vA + vB), f"Wrong operation from source 'd'")
        self.assertListEqual(list(df['var_1']), list(vA + vB + n1), f"Wrong operation from 'var_1'")
        self.assertListEqual(list(df['var_2']), list(vA + vB), f"Wrong operation from 'var_2'")
        self.assertListEqual(list(df['var_3']), list(vA + vB + n3), f"Wrong operation from 'var_3'")
        # create wrong descendants
        with self.assertRaises(AssertionError):
            dg.descendant(lambda a, b: a + b, name='c')
        with self.assertRaises(AssertionError):
            dg.descendant(lambda a, b, ext: a + b + ext)
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y: x + y, name='c', parents=[A, B])
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y: x + y, parents=[A, B, Ext])
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y, z: x + y + z, parents=[A, B])
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y, z: x + y + z, parents=[A, B, Ext])
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y: x + y, name='c', parents=['a', 'b'])
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y: x + y, parents=['a', 'b', 'ext'])
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y, z: x + y + z, parents=['a', 'b'])
        with self.assertRaises(AssertionError):
            dg.descendant(lambda x, y, z: x + y + z, parents=['a', 'b', 'ext'])
        with self.assertRaises(AssertionError):
            dg.descendant(A + B, name='c')
        with self.assertRaises(AssertionError):
            dg.descendant(A + B, parents=[A, B])
        with self.assertRaises(AssertionError):
            dg.descendant(A + B, parents=['a', 'b'])
        with self.assertRaises(AssertionError):
            dg.descendant(A + B + Ext)

    # noinspection PyPep8Naming
    def test_generation(self):
        dg, rand = Generator(seed=0), np.random.default_rng(0)
        # correct values
        h = rand.uniform(size=size)
        z = rand.binomial(n=1, p=0.5, size=size)
        noise_1 = rand.normal(size=size)
        noise_2 = rand.normal(size=size)
        x = (2 * z - 1) * h + 0.01 * noise_1
        y = (x ** 2) * (0.01 * noise_2 + 1)
        # generator nodes
        H = dg.uniform(hidden=True, name='h')
        Z = dg.binomial(hidden=False, name='z')
        X = dg.descendant((2 * Z - 1) * H + 0.01 * dg.noise(), name='x')
        dg.descendant((X ** 2) * (0.01 * dg.noise() + 1), name='y')
        # check that generated values are correct
        df = dg.generate(num=size, hidden=True)
        self.assertTrue(np.allclose(df['h'], h), f"Wrong values for 'h'")
        self.assertTrue(np.allclose(df['z'], z), f"Wrong values for 'z'")
        self.assertTrue(np.allclose(df['noise_1'], noise_1), f"Wrong values for 'noise_1'")
        self.assertTrue(np.allclose(df['noise_2'], noise_2), f"Wrong values for 'noise_2'")
        self.assertTrue(np.allclose(df['x'], x), f"Wrong values for 'x'")
        self.assertTrue(np.allclose(df['y'], y), f"Wrong values for 'y'")
        # check that hidden=False returns visible nodes only
        df = dg.generate(num=size, hidden=False)
        self.assertListEqual(['z', 'x', 'y'], list(df.columns), f"Wrong nodes returned by generation")
