from typing import Dict, Tuple

import numpy as np

SIZE = 10

RNG = np.random.default_rng(0)

UNARY_OPERATIONS: Dict[str, np.ndarray] = {
    'invert': RNG.integers(-100, 100, size=SIZE),
    'negative': RNG.uniform(-100, 100, size=SIZE),
    'abs': RNG.uniform(-100, 100, size=SIZE),
    'floor': RNG.uniform(-100, 100, size=SIZE),
    'ceil': RNG.uniform(-100, 100, size=SIZE),
    'trunc': RNG.uniform(-100, 100, size=SIZE),
    'round': RNG.uniform(-100, 100, size=SIZE),
    'sin': RNG.uniform(-100, 100, size=SIZE),
    'cos': RNG.uniform(-100, 100, size=SIZE),
    'tan': RNG.uniform(-100, 100, size=SIZE),
    'arcsin': RNG.uniform(-1, 1, size=SIZE),
    'arccos': RNG.uniform(-1, 1, size=SIZE),
    'arctan': RNG.uniform(-100, 100, size=SIZE),
    'sinh': RNG.uniform(-100, 100, size=SIZE),
    'cosh': RNG.uniform(-100, 100, size=SIZE),
    'tanh': RNG.uniform(-100, 100, size=SIZE),
    'arcsinh': RNG.uniform(-100, 100, size=SIZE),
    'arccosh': RNG.uniform(1, 100, size=SIZE),
    'arctanh': RNG.uniform(-1, 1, size=SIZE),
    'exp': RNG.uniform(-100, 100, size=SIZE),
    'exp2': RNG.uniform(-100, 100, size=SIZE),
    'expm1': RNG.uniform(-100, 100, size=SIZE),
    'log': RNG.uniform(0, 100, size=SIZE),
    'log2': RNG.uniform(0, 100, size=SIZE),
    'log10': RNG.uniform(0, 100, size=SIZE),
    'log1p': RNG.uniform(-1, 100, size=SIZE),
    'square': RNG.uniform(-100, 100, size=SIZE),
    'cbrt': RNG.uniform(-100, 100, size=SIZE),
}

BINARY_OPERATIONS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    'add': (RNG.uniform(-100, 100, size=SIZE), RNG.uniform(-100, 100, size=SIZE)),
    'subtract': (RNG.uniform(-100, 100, size=SIZE), RNG.uniform(-100, 100, size=SIZE)),
    'multiply': (RNG.uniform(-100, 100, size=SIZE), RNG.uniform(-100, 100, size=SIZE)),
    'true_divide': (RNG.uniform(1, 100, size=SIZE), RNG.uniform(1, 100, size=SIZE)),
    'floor_divide': (RNG.integers(1, 100, size=SIZE), RNG.integers(1, 100, size=SIZE)),
    'power': (RNG.uniform(1, 100, size=SIZE), RNG.uniform(1, 100, size=SIZE)),
    'mod': (RNG.integers(1, 100, size=SIZE), RNG.integers(1, 100, size=SIZE)),
    'bitwise_and': (RNG.integers(0, 2, size=SIZE), RNG.integers(0, 2, size=SIZE)),
    'bitwise_or': (RNG.integers(0, 2, size=SIZE), RNG.integers(0, 2, size=SIZE)),
    'bitwise_xor': (RNG.integers(0, 2, size=SIZE), RNG.integers(0, 2, size=SIZE))
}

DISTRIBUTIONS: Dict[str, Tuple[dict, dict]] = {
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

ALIASES: Dict[str, Tuple[str, dict]] = {
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
    'custom': ('var', dict(distribution=lambda s: RNG.random(size=s)))
}
