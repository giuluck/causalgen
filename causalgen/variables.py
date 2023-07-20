from abc import abstractmethod, ABC
from typing import Callable, Optional, Any, List

import numpy as np


class Variable(ABC):
    """
    Interface for a generator node or intermediate operation.

    Each variable has an internal value which is defined as a function of its parents. Additionally, they implement
    basic arithmetic and logical operations which return can either operate inplace or return other variables.
    """

    @property
    @abstractmethod
    def value(self) -> np.ndarray:
        """The current value of the variable.

        :return:
            A numpy array computed as a function of the variable's parents.
        """
        pass

    @property
    @abstractmethod
    def parents(self) -> list:
        """The parent of the variable.

        :return:
            A list of Variable objects which represent the variable's parents.
        """
        pass

    @abstractmethod
    def _operation(self, operator: Callable) -> Any:
        """Performs a unary operation on the variable.

        :param operator:
            The name of the unary operator, which must be a numpy operator.

        :return:
            A Variable object with the operation result.
        """
        pass

    def _left_operation(self, other: Any, operator: Callable) -> Any:
        """Performs a binary operation on the variable when the variable is on the left-hand side.

        :param other:
            The other object instance, which may be either another Variable or a different type.

        :param operator:
            The name of the binary operator, which must be a numpy operator.

        :return:
            A Variable object with the operation result.
        """
        if isinstance(other, Variable):
            return Intermediate(op=operator, inputs=[self, other])
        else:
            return self._operation(operator=lambda s: operator(s, other))

    def _right_operation(self, other: Any, operator: Callable) -> Any:
        """Performs a binary operation on the variable when the variable is on the right-hand side.

        :param other:
            The other object instance, which may be either another Variable or a different type.

        :param operator:
            The name of the binary operator, which must be a numpy operator.

        :return:
            A Variable object with the operation result.
        """
        if isinstance(other, Variable):
            return Intermediate(op=operator, inputs=[other, self])
        else:
            return self._operation(operator=lambda s: operator(other, s))

    def __add__(self, other):
        return self._left_operation(other, operator=np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._left_operation(other, operator=np.subtract)

    def __rsub__(self, other):
        return self._right_operation(other, operator=np.subtract)

    def __mul__(self, other):
        # allows to handle np.square as a unary operation since the numpy method calls __mul__ instead of __pow__
        if other is self:
            return self._operation(operator=lambda s: np.multiply(s, s))
        return self._left_operation(other, operator=np.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._left_operation(other, operator=np.true_divide)

    def __rtruediv__(self, other):
        return self._right_operation(other, operator=np.true_divide)

    def __floordiv__(self, other):
        return self._left_operation(other, operator=np.floor_divide)

    def __rfloordiv__(self, other):
        return self._right_operation(other, operator=np.floor_divide)

    def __pow__(self, power):
        return self._left_operation(power, operator=np.power)

    def __rpow__(self, other):
        return self._right_operation(other, operator=np.power)

    def __mod__(self, other):
        return self._left_operation(other, operator=np.mod)

    def __rmod__(self, other):
        return self._right_operation(other, operator=np.mod)

    def __and__(self, other):
        return self._left_operation(other, operator=np.bitwise_and)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return self._left_operation(other, operator=np.bitwise_or)

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        return self._left_operation(other, operator=np.bitwise_xor)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __invert__(self):
        return self._operation(operator=np.invert)

    def __neg__(self):
        return self._operation(operator=np.negative)

    def __abs__(self):
        return self._operation(operator=np.abs)

    def __floor__(self):
        return self._operation(operator=np.floor)

    def __ceil__(self):
        return self._operation(operator=np.ceil)

    def __trunc__(self):
        return self._operation(operator=np.trunc)

    def rint(self):
        return self._operation(operator=np.rint)

    def sin(self):
        return self._operation(operator=np.sin)

    def cos(self):
        return self._operation(operator=np.cos)

    def tan(self):
        return self._operation(operator=np.tan)

    def arcsin(self):
        return self._operation(operator=np.arcsin)

    def arccos(self):
        return self._operation(operator=np.arccos)

    def arctan(self):
        return self._operation(operator=np.arctan)

    def sinh(self):
        return self._operation(operator=np.sinh)

    def cosh(self):
        return self._operation(operator=np.cosh)

    def tanh(self):
        return self._operation(operator=np.tanh)

    def arcsinh(self):
        return self._operation(operator=np.arcsinh)

    def arccosh(self):
        return self._operation(operator=np.arccosh)

    def arctanh(self):
        return self._operation(operator=np.arctanh)

    def exp(self):
        return self._operation(operator=np.exp)

    def exp2(self):
        return self._operation(operator=np.exp2)

    def expm1(self):
        return self._operation(operator=np.expm1)

    def log(self):
        return self._operation(operator=np.log)

    def log2(self):
        return self._operation(operator=np.log2)

    def log10(self):
        return self._operation(operator=np.log10)

    def log1p(self):
        return self._operation(operator=np.log1p)

    def cbrt(self):
        return self._operation(operator=np.cbrt)


class Intermediate(Variable):
    """
    A variable which contains the intermediate results of other variable's operations.

    It contains an ordered list of parents which are used as inputs to retrieve the value of the variable according to
    a given base operation, and a list of unary operations which are then applied to this intermediate result.
    When an intermediate variable is part of a binary operation, a new intermediate variable is generated and returned.
    """

    def __init__(self, op: Callable[..., np.ndarray], inputs: List[Variable]):
        """
        :param op:
            The base operation to be applied to the list of inputs.

        :param inputs:
            An ordered list of parents to be used as inputs for the base operation.
        """
        self._inputs: List[Variable] = inputs
        self._base_op: Callable[..., np.ndarray] = op
        self._unary_ops: List[Callable[[np.ndarray], np.ndarray]] = []

    def _operation(self, operator: Callable) -> Variable:
        # unary operations on intermediate variables operate inplace instead of returning a new variable
        # the list of unary operations are appended to a list until the variable is part of a binary operation
        self._unary_ops.append(operator)
        return self

    @property
    def value(self) -> np.ndarray:
        x = self._base_op(*[inp.value for inp in self._inputs])
        for op in self._unary_ops:
            x = op(x)
        return x

    @property
    def parents(self) -> list:
        return list(self._inputs)

    def __repr__(self) -> str:
        location = super(Variable, self).__repr__().split(' ')[-1][:-1]
        return f"IntermediateVariable('{location}')"


class Node(Variable):
    """
    A variable which is part of a Generator object.

    It contains a name (which must be unique in the generator) and other information such as whether it is a hidden
    variable in the causal graph, its set of parents, its generative function, and a reference to the generator
    instance it belongs to.
    """

    def __init__(self, generator, func: Callable[[], np.ndarray], parents: set, name: str, hidden: bool):
        """
        :param generator:
            Reference to the generator object that contains the node.

        :param func:
            The function used to generate the node's value.

        :param parents:
            The set of node's parents.

        :param name:
            The name of the node.

        :param hidden:
            Whether the node represents a hidden variable in the causal graph or not.
        """
        self._name: str = name
        self._hidden: bool = hidden
        self._parents: set = parents
        self._generator: Any = generator
        self._func: Callable[[], np.ndarray] = func
        self._value: Optional[np.ndarray] = None

    def _operation(self, operator: Callable) -> Any:
        return Intermediate(op=operator, inputs=[self])

    @property
    def value(self) -> Optional[np.ndarray]:
        """The current value of the node.

        The value is stored after the node is sampled using the 'sample()' method, and then removed once the 'clear()'
        method is called. There are no sanity checks on the value, hence if this property is called before the node is
        sampled, the returned value will be None.
        """
        return self._value

    @property
    def parents(self) -> list:
        """The (unordered) list of node's parents"""
        return list(self._parents)

    @property
    def source(self) -> bool:
        """Whether the node is a source (i.e., it has no parents) or not."""
        return len(self._parents) == 0

    @property
    def name(self) -> str:
        """The name of the node."""
        return self._name

    @property
    def generator(self) -> Any:
        """The generator instance that contains the node."""
        return self._generator

    @property
    def visible(self) -> bool:
        """Whether the node represents a visible variable in the causal graph."""
        return not self._hidden

    @property
    def hidden(self) -> bool:
        """Whether the node represents a hidden variable in the causal graph."""
        return self._hidden

    def sample(self) -> np.ndarray:
        """Samples a new value for the node.

        In order to sample the node, its current value must be not assigned (i.e., None), otherwise an exception is
        raised. Moreover, since the node needs its parents' values in order to be sampled, an exception is raised if at
        least one of the parents has not been sampled yet (i.e., if their value is None).

        :return:
            The value of the node after it is sampled.
        """
        assert self._value is None, f"Node '{self.name}' must be clear before sampling it, but value is not None"
        for par in self._parents:
            assert par.value is not None, f"Parent '{par.name}' has not been assigned yet"
        self._value = self._func()
        return self._value

    def clear(self):
        """Clears the value of the node.

        If the node's value is already unassigned (i.e., it is None), raises an exception.
        """
        assert self._value is not None, f"Variable is already clean, its value is None"
        self._value = None

    def __repr__(self) -> str:
        return f"Node('{self._name}', parents={[p.name for p in self._parents]}, hidden={self._hidden})"
