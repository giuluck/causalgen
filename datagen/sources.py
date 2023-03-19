from typing import Optional, Callable, Any

import numpy as np

import datagen
from datagen.variables import Intermediate, Variable


class Size(Variable):
    @property
    def parents(self) -> list:
        raise TypeError("Singleton variable 'SIZE' has no parents")

    def _unary_operation(self, operator: Callable) -> Any:
        raise TypeError("Singleton variable 'SIZE' cannot be part of any operation")

    def _binary_operation(self, other: Any, operator: Callable) -> Any:
        raise TypeError("Singleton variable 'SIZE' cannot be part of any operation")

    def __init__(self):
        self._value: Optional[np.ndarray] = None

    @property
    def value(self) -> np.ndarray:
        return self._value

    def set(self, size: int):
        assert self._value is None, f"{self} variable is already being used, its value is not None as expected"
        self._value = np.array(size)

    def clear(self) -> Any:
        assert self._value is not None, f"{self} variable is already clean, its value is None but it should not be"
        self._value = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Size, cls).__new__(cls)
        return getattr(cls, 'instance')

    def __repr__(self):
        return 'Singleton(SIZE)'


SIZE = Size()


def noise(amount: float = 1.0) -> Variable:
    op = lambda: datagen.random.normal(loc=0.0, scale=amount, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def uniform(low: float = 0.0, high: float = 1.0) -> Variable:
    op = lambda: datagen.random.uniform(low=low, high=high, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def normal(loc: float = 0.0, scale: float = 1.0) -> Variable:
    op = lambda: datagen.random.normal(loc=loc, scale=scale, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def lognormal(mean: float = 0.0, sigma: float = 1.0) -> Variable:
    op = lambda: datagen.random.lognormal(mean=mean, sigma=sigma, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def exponential(scale: float = 1.0) -> Variable:
    op = lambda: datagen.random.exponential(scale=scale)
    return Intermediate(op=op, inputs=[])


def poisson(lam: float = 1.0) -> Variable:
    op = lambda: datagen.random.poisson(lam=lam, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def geometric(p: float = 0.5) -> Variable:
    op = lambda: datagen.random.geometric(p=p, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def binomial(p: float = 0.5) -> Variable:
    op = lambda: datagen.random.binomial(n=1, p=p, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


# noinspection PyUnresolvedReferences
def integers(low: int = 0, high: float = 1, endpoint: bool = True) -> Variable:
    # behaviour explained in: https://numpy.org/doc/1.21/reference/random/generated/numpy.random.Generator.integers.html
    if isinstance(datagen.random, np.random.Generator):
        op = lambda: datagen.random.integers(low=low, high=high, endpoint=endpoint, size=SIZE.value)
    elif endpoint:
        op = lambda: np.random.random_integers(low, high, size=SIZE.value)
    else:
        op = lambda: np.random.randint(low, high, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def choice(choices: list, replace: bool = True, p: Optional[list] = None) -> Variable:
    op = lambda: datagen.random.choice(a=choices, replace=replace, p=p, size=SIZE.value)
    return Intermediate(op=op, inputs=[])


def custom(distribution: Callable[[int], np.ndarray]) -> Variable:
    op = lambda: distribution(SIZE.value)
    return Intermediate(op=op, inputs=[])
