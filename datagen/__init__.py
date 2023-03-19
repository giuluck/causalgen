import numpy as np

from datagen.generator import Generator
from datagen.sources import noise, uniform, normal, lognormal, exponential, poisson, geometric, binomial, integers, \
    choice, custom

random: np.random.Generator = np.random


def set_seed(seed) -> np.random.Generator:
    globals()['random'] = np.random if seed is None else np.random.default_rng(seed)
    return random
