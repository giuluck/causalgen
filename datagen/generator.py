import inspect
from typing import Callable, Optional, Dict, Union, List, Set, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from datagen.variables import Variable, Node, Intermediate


class Generator:
    @staticmethod
    def _get_rng(seed: Union[None, int, np.random.Generator]) -> np.random.Generator:
        if seed is None:
            # noinspection PyTypeChecker
            return np.random
        elif isinstance(seed, int):
            return np.random.default_rng(seed)
        else:
            return seed

    def __init__(self, seed: Union[None, int, np.random.Generator] = 42):
        self._rng: np.random.Generator = Generator._get_rng(seed)
        self._size: Optional[int] = None
        self._counters: Dict[str, int] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, Node] = {}

    @property
    def random(self) -> np.random.Generator:
        return self._rng

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph.copy()

    @property
    def nodes(self) -> List[Node]:
        return [v for v in self._nodes.values()]

    @property
    def hidden(self) -> List[Node]:
        return [v for v in self._nodes.values() if v.hidden]

    @property
    def visible(self) -> List[Node]:
        return [v for v in self._nodes.values() if v.visible]

    @property
    def sources(self) -> List[Node]:
        return [v for v in self._nodes.values() if v.source]

    def reset_seed(self, seed: Union[None, int, np.random.Generator]) -> Any:
        self._rng = Generator._get_rng(seed)
        return self

    def noise(self, amount: float = 1.0, hidden: bool = True, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.normal(loc=0.0, scale=amount, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='noise')

    def uniform(self, low: float = 0.0, high: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.uniform(low=low, high=high, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='uni')

    def normal(self, mu: float = 0.0, sigma: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.normal(loc=mu, scale=sigma, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='norm')

    def lognormal(self, mu: float = 0.0, sigma: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.lognormal(mean=mu, sigma=sigma, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='lnorm')

    def exponential(self, scale: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.exponential(scale=scale, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='exp')

    def poisson(self, lam: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.poisson(lam=lam, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='pois')

    def geometric(self, p: float = 0.5, hidden: bool = False, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.geometric(p=p, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='geom')

    def binomial(self, p: float = 0.5, hidden: bool = False, name: Optional[str] = None) -> Node:
        func = lambda: self._rng.binomial(n=1, p=p, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='bin')

    def integers(self,
                 low: int = 0,
                 high: int = 1,
                 endpoint: bool = True,
                 hidden: bool = False,
                 name: Optional[str] = None) -> Node:
        # refer to: https://numpy.org/doc/1.21/reference/random/generated/numpy.random.Generator.integers.html
        if isinstance(self._rng, np.random.Generator):
            func = lambda: self._rng.integers(low=low, high=high, endpoint=endpoint, size=self._size)
        elif endpoint:
            func = lambda: np.random.random_integers(low, high, size=self._size)
        else:
            func = lambda: np.random.randint(low, high, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='int')

    def choice(self,
               categories: list,
               replace: bool = True,
               p: Optional[list] = None,
               hidden: bool = False,
               name: Optional[str] = None) -> Node:
        func = lambda: self._rng.choice(a=categories, replace=replace, p=p, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='cat')

    def custom(self,
               distribution: Callable[[int], np.ndarray],
               hidden: bool = False,
               name: Optional[str] = None) -> Node:
        # check that the distribution signature matches the expected one (i.e., f(int) -> array)
        signature = inspect.signature(distribution).parameters.keys()
        assert len(signature) == 1, f"Custom distributions should match signature f(size: int) -> np.ndarray"
        func = lambda: distribution(self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist=None)

    def descendant(self,
                   function: Union[Variable, Callable],
                   parents: Optional[List[Union[str, Node]]] = None,
                   name: Optional[str] = None,
                   hidden: bool = False) -> Node:
        if isinstance(function, Variable):
            assert parents is None, f"Derived variables must have no parents, please use 'parents=None'"
            return self._add_variable_descendant(var=function, name=name, hidden=hidden)
        elif isinstance(function, Callable):
            return self._add_function_descendant(func=function, name=name, hidden=hidden, parents=parents)
        else:
            raise TypeError(f"Unsupported function type '{type(function).__name__}'")

    def _add_variable_descendant(self, var: Variable, name: Optional[str], hidden: bool) -> Node:
        if isinstance(var, Intermediate):
            def _retrieve_parents(v: Intermediate) -> Set[Node]:
                output = set()
                for p in v.parents:
                    if isinstance(p, Intermediate):
                        output.update(_retrieve_parents(p))
                    elif isinstance(p, Node):
                        output.add(p)
                    else:
                        raise TypeError(f"Unknown variable type '{type(v).__name__}'")
                return output

            parents = _retrieve_parents(var)
            return self._check_node_and_append(func=lambda: var.value, parents=parents, hidden=hidden, name=name)
        elif isinstance(var, Node):
            # otherwise, the new node is simply an identity function from the previous one
            return self._check_node_and_append(func=lambda: var.value, name=name, hidden=hidden, parents={var})
        else:
            raise TypeError(f"Unknown variable type '{type(var).__name__}'")

    def _add_function_descendant(self,
                                 func: Callable[..., np.ndarray],
                                 name: Optional[str],
                                 hidden: bool,
                                 parents: Optional[List[Union[str, Node]]]) -> Node:
        signature = inspect.signature(func).parameters.keys()
        # if no parent is passed, their names are assumed by the function input parameters, otherwise check consistency
        if parents is None:
            parents = list(signature)
        else:
            assert len(signature) == len(parents), f"Number of input parameters do not match number of parents"
        # for parents that are passed via name, check that the name is present in the structure and retrieve the node
        inputs = []
        for par in parents:
            if isinstance(par, str):
                parent = self._nodes.get(par)
                assert parent is not None, f"Unknown parent '{par}'"
            else:
                parent = par
            inputs.append(parent)
        # eventually build the node
        function = lambda: func(*[inp.value for inp in inputs])
        return self._check_node_and_append(func=function, name=name, hidden=hidden, parents=set(inputs), dist=None)

    def _check_node_and_append(self,
                               func: Callable[..., np.ndarray],
                               name: Optional[str],
                               parents: Set[Node],
                               hidden: Optional[bool],
                               dist: Optional[str] = None) -> Node:
        # assign default name if no name is passed, otherwise check for name conflicts with existing variables
        if name is None:
            prefix = 'var' if dist is None else dist
            self._counters[prefix] = self._counters.get(prefix, 0)
            self._counters[prefix] += 1
            name = f'{prefix}_{self._counters[prefix]}'
        else:
            assert name not in self._nodes, f"Name '{name}' has already been assigned to a variable"
        # check that parents belong to this generator
        for par in parents:
            assert par.generator is self, f"Parent '{par.name}' does not belong to this generator"
        # build node variable and add it to the internal data structures
        node = Node(generator=self, func=func, name=name, hidden=hidden, parents=parents)
        self._nodes[name] = node
        self._graph.add_node(name)
        for par in node.parents:
            self._graph.add_edge(par.name, name)
        # check graph consistency and return node
        assert nx.is_directed_acyclic_graph(self._graph), "The resulting causal graph should be a DAG but it is no more"
        return node

    def generate(self, num: int = 1, hidden: bool = False) -> pd.DataFrame:
        data = pd.DataFrame()
        assert self._size is None, "Unexpected behaviour, internal variable '_size' should be None but it is not"
        self._size = num
        for node in nx.topological_sort(self._graph):
            data[node] = self._nodes[node].sample()
        for node in self._nodes.values():
            node.clear()
        self._size = None
        return data if hidden else data[[n.name for n in self.visible]]

    def visualize(self,
                  fig_size: Tuple[int, int] = (10, 6),
                  node_size: float = 10,
                  arrow_size: float = 8,
                  edge_width: float = 1):
        g = self._graph.copy()
        for layer, nodes in enumerate(nx.topological_generations(g)):
            for node in nodes:
                g.nodes[node]['layer'] = layer
        pos = nx.multipartite_layout(g, subset_key='layer')
        plt.figure(figsize=fig_size)
        nx.draw(
            g,
            pos=pos,
            node_color=['#3466AA' if n.visible else '#C0C0C0' for n in self.nodes],
            node_size=node_size * 100,
            linewidths=edge_width,
            arrowsize=arrow_size,
            width=edge_width,
            with_labels=True,
            arrows=True
        )
        plt.gca().collections[0].set_edgecolor("#000000")
        plt.show()
