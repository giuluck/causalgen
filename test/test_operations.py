import unittest

import numpy as np

from causalgen.variables import Intermediate, Node
from test.test_utils import UNARY_OPERATIONS, BINARY_OPERATIONS


class TestOperations(unittest.TestCase):
    def test_intermediate(self):
        # TEST UNARY OPERATIONS
        for op, vec in UNARY_OPERATIONS.items():
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
        for op, (vec1, vec2) in BINARY_OPERATIONS.items():
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
        for op, (vec1, vec2) in BINARY_OPERATIONS.items():
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
        for op, (vec1, vec2) in BINARY_OPERATIONS.items():
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

    def test_node(self):
        # TEST UNARY OPERATIONS
        for op, vec in UNARY_OPERATIONS.items():
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
        for op, (vec1, vec2) in BINARY_OPERATIONS.items():
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
        for op, (vec1, vec2) in BINARY_OPERATIONS.items():
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
        for op, (vec1, vec2) in BINARY_OPERATIONS.items():
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
