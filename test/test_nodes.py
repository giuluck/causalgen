import unittest

import numpy as np

from causalgen import Generator
from causalgen.variables import Node
from test.test_utils import ALIASES, SIZE


class TestNodes(unittest.TestCase):
    def test_sources(self):
        dg = Generator()
        for operator, (alias, kwargs) in ALIASES.items():
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
        vA = rand.normal(size=SIZE)
        self.assertIs(A.hidden, False, f"Node A should be visible")
        self.assertIs(A.visible, True, f"Node A should be visible")
        B = dg.integers(hidden=True, name='b')
        vB = rand.integers(0, 1, endpoint=True, size=SIZE)
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
        df = dg.generate(num=SIZE, hidden=True)
        n1, n3 = rand.normal(scale=0.1, size=SIZE), rand.normal(scale=0.1, size=SIZE)
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
