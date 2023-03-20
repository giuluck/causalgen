from abc import abstractmethod, ABC
from typing import Callable, Optional, Any, List

import numpy as np


class Variable(ABC):
    @property
    @abstractmethod
    def value(self, *args: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def parents(self) -> list:
        pass

    @abstractmethod
    def _unary_operation(self, operator: Callable) -> Any:
        pass

    @abstractmethod
    def _binary_operation(self, other: Any, operator: Callable) -> Any:
        pass

    @abstractmethod
    def _right_binary_operation(self, other: Any, operator: Callable) -> Any:
        pass

    @abstractmethod
    def square(self):
        # allows to handle np.square as a unary operation
        pass

    def __add__(self, other):
        return self._binary_operation(other, operator=np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_operation(other, operator=np.subtract)

    def __rsub__(self, other):
        return self._right_binary_operation(other, operator=np.subtract)

    def __mul__(self, other):
        # allows to handle np.square as a unary operation
        if other is self:
            return self.square()
        return self._binary_operation(other, operator=np.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_operation(other, operator=np.true_divide)

    def __rtruediv__(self, other):
        return self._right_binary_operation(other, operator=np.true_divide)

    def __floordiv__(self, other):
        return self._binary_operation(other, operator=np.floor_divide)

    def __rfloordiv__(self, other):
        return self._right_binary_operation(other, operator=np.floor_divide)

    def __pow__(self, power):
        return self._binary_operation(power, operator=np.power)

    def __rpow__(self, other):
        return self._right_binary_operation(other, operator=np.power)

    def __mod__(self, other):
        return self._binary_operation(other, operator=np.mod)

    def __rmod__(self, other):
        return self._right_binary_operation(other, operator=np.mod)

    def __and__(self, other):
        return self._binary_operation(other, operator=np.bitwise_and)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return self._binary_operation(other, operator=np.bitwise_or)

    def __ror__(self, other):
        return self.__or__(other)

    def __xor__(self, other):
        return self._binary_operation(other, operator=np.bitwise_xor)

    def __rxor__(self, other):
        return self.__xor__(other)

    def __invert__(self):
        return self._unary_operation(operator=np.invert)

    def __neg__(self):
        return self._unary_operation(operator=np.negative)

    def __abs__(self):
        return self._unary_operation(operator=np.abs)

    def __floor__(self):
        return self._unary_operation(operator=np.floor)

    def __ceil__(self):
        return self._unary_operation(operator=np.ceil)

    def __trunc__(self):
        return self._unary_operation(operator=np.trunc)

    def rint(self):
        return self._unary_operation(operator=np.rint)

    def sin(self):
        return self._unary_operation(operator=np.sin)

    def cos(self):
        return self._unary_operation(operator=np.cos)

    def tan(self):
        return self._unary_operation(operator=np.tan)

    def arcsin(self):
        return self._unary_operation(operator=np.arcsin)

    def arccos(self):
        return self._unary_operation(operator=np.arccos)

    def arctan(self):
        return self._unary_operation(operator=np.arctan)

    def sinh(self):
        return self._unary_operation(operator=np.sinh)

    def cosh(self):
        return self._unary_operation(operator=np.cosh)

    def tanh(self):
        return self._unary_operation(operator=np.tanh)

    def arcsinh(self):
        return self._unary_operation(operator=np.arcsinh)

    def arccosh(self):
        return self._unary_operation(operator=np.arccosh)

    def arctanh(self):
        return self._unary_operation(operator=np.arctanh)

    def exp(self):
        return self._unary_operation(operator=np.exp)

    def exp2(self):
        return self._unary_operation(operator=np.exp2)

    def expm1(self):
        return self._unary_operation(operator=np.expm1)

    def log(self):
        return self._unary_operation(operator=np.log)

    def log2(self):
        return self._unary_operation(operator=np.log2)

    def log10(self):
        return self._unary_operation(operator=np.log10)

    def log1p(self):
        return self._unary_operation(operator=np.log1p)

    def cbrt(self):
        return self._unary_operation(operator=np.cbrt)


class Intermediate(Variable):
    def __init__(self, op: Callable[..., np.ndarray], inputs: List[Variable]):
        self._inputs: List[Variable] = inputs
        self._base_op: Callable[..., np.ndarray] = op
        self._unary_ops: List[Callable[[np.ndarray], np.ndarray]] = []

    def _unary_operation(self, operator: Callable) -> Variable:
        self._unary_ops.append(lambda x: operator(x))
        return self

    def _binary_operation(self, other: Any, operator: Callable) -> Any:
        if isinstance(other, Variable):
            return Intermediate(op=lambda s, o: operator(s.value, o.value), inputs=[self, other])
        else:
            self._unary_ops.append(lambda x: operator(x, other))
            return self

    def _right_binary_operation(self, other: Any, operator: Callable) -> Any:
        if isinstance(other, Variable):
            return Intermediate(op=lambda s, o: operator(o.value, s.value), inputs=[self, other])
        else:
            self._unary_ops.append(lambda x: operator(other, x))
            return self

    def square(self):
        self._unary_ops.append(lambda x: x * x)
        return self

    @property
    def value(self) -> np.ndarray:
        x = self._base_op(*self._inputs)
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
    def __init__(self, generator, func: Callable[[], np.ndarray], parents: set, name: str, hidden: bool):
        self._name: str = name
        self._hidden: bool = hidden
        self._parents: set = parents
        self._generator: Any = generator
        self._func: Callable[[], np.ndarray] = func
        self._value: Optional[np.ndarray] = None
        super(Node, self).__init__()

    def _unary_operation(self, operator: Callable) -> Any:
        return Intermediate(op=lambda s: operator(s.value), inputs=[self])

    def _binary_operation(self, other: Any, operator: Callable) -> Any:
        if isinstance(other, Variable):
            return Intermediate(op=lambda s, o: operator(s.value, o.value), inputs=[self, other])
        else:
            return Intermediate(op=lambda s: operator(s.value, other), inputs=[self])

    def _right_binary_operation(self, other: Any, operator: Callable) -> Any:
        if isinstance(other, Variable):
            return Intermediate(op=lambda s, o: operator(o.value, s.value), inputs=[self, other])
        else:
            return Intermediate(op=lambda s: operator(other, s.value), inputs=[self])

    def square(self):
        return Intermediate(op=lambda s: s.value * s.value, inputs=[self])

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def parents(self) -> list:
        return list(self._parents)

    @property
    def source(self) -> bool:
        return len(self._parents) == 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def generator(self) -> Any:
        return self._generator

    @property
    def visible(self) -> bool:
        return not self._hidden

    @property
    def hidden(self) -> bool:
        return self._hidden

    def sample(self) -> np.ndarray:
        assert self._value is None, f"Node '{self.name}' must be clear before sampling it, but value is not None"
        for par in self._parents:
            assert par.value is not None, f"Parent '{par.name}' has not been assigned yet"
        self._value = self._func()
        return self._value

    def clear(self) -> Any:
        assert self._value is not None, f"Variable is already clean, its value is None"
        self._value = None
        return self

    def __repr__(self) -> str:
        parents = [par.name for par in self._parents]
        return f"Node('{self._name}', parents={parents}, hidden={self._hidden})"
