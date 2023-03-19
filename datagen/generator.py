import inspect
from typing import Callable, Optional, Dict, Union, List, Set

import networkx as nx
import numpy as np
import pandas as pd
from descriptors import classproperty
from matplotlib import pyplot as plt

import datagen
from datagen.sources import SIZE
from datagen.variables import Variable, Node, Intermediate


class Generator:
    def __init__(self):
        self._counter: int = 0
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, Node] = {}

    @classproperty
    def rng(self) -> np.random.Generator:
        return datagen.random

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

    def add(self,
            function: Union[str, Variable, Callable],
            parents: Optional[List[Union[str, Node]]] = None,
            name: Optional[str] = None,
            hidden: bool = False,
            **kwargs) -> Node:
        if isinstance(function, str):
            assert parents is None, f"Numpy distributions must have no parents, please use 'parents=None'"
            return self._add_source(dist=function, name=name, hidden=hidden, **kwargs)
        elif isinstance(function, Variable):
            assert len(kwargs) == 0, f"Derived variables must have no additional arguments, please use no kwargs"
            assert parents is None, f"Derived variables must have no parents, please use 'parents=None'"
            return self._add_derived(var=function, name=name, hidden=hidden)
        elif isinstance(function, Callable):
            assert len(kwargs) == 0, f"Custom variables must have no additional arguments, please use no kwargs"
            return self._add_custom(func=function, name=name, hidden=hidden, parents=parents)
        else:
            raise TypeError(f"Unsupported function type '{type(function).__name__}'")

    def _add_source(self, dist: str, name: Optional[str], hidden: bool, **kwargs) -> Node:
        if dist == 'noise':
            # for noise distributions, check that a single parameter is passed (i.e., amount)
            amount = kwargs.get('amount')
            assert amount is not None, f"Noise distributions admit a single parameter ('amount')"
            assert len(kwargs) == 1, f"Noise distributions admit a single parameter ('amount')"
            dist = lambda: self.rng.normal(loc=0.0, scale=kwargs['amount'], size=SIZE.value)
        elif dist == 'custom':
            # for custom distributions, check that a single parameter is passed (i.e., distribution)
            # the distribution must be a function whose signature matches the expected one (i.e., f(int) -> array)
            dist = kwargs.get('distribution')
            assert dist is not None, f"Custom distributions admit a single parameter ('distribution')"
            assert len(kwargs) == 1, f"Custom distributions admit a single parameter ('distribution')"
            signature = inspect.signature(dist).parameters.keys()
            assert len(signature) == 1, f"Custom distributions should match signature f(size: int) -> np.ndarray"
        else:
            # otherwise retrieve the distribution from the random number generator and build the function accordingly
            assert hasattr(self.rng, dist), f"Unknown source distribution '{dist}'"
            function = getattr(self.rng, dist)
            dist = lambda: function(size=SIZE.value, **kwargs)
        return self._check_node_and_append(func=dist, name=name, hidden=hidden, parents=set())

    def _add_derived(self, var: Variable, name: Optional[str], hidden: bool) -> Node:
        if isinstance(var, Intermediate):
            def _retrieve_parents(v: Intermediate) -> Set[Node]:
                output = set()
                for p in v.parents:
                    if isinstance(p, Intermediate):
                        output.update(_retrieve_parents(p))
                    elif isinstance(p, Node):
                        output.add(p)
                    elif p is not SIZE:
                        raise TypeError(f"Unknown variable type '{type(v).__name__}'")
                return output

            parents = _retrieve_parents(var)
            return self._check_node_and_append(func=lambda: var.value, parents=parents, hidden=hidden, name=name)
        elif isinstance(var, Node):
            # otherwise, the new node is simply an identity function from the previous one
            return self._check_node_and_append(func=lambda: var.value, name=name, hidden=hidden, parents={var})
        else:
            raise TypeError(f"Unknown variable type '{type(var).__name__}'")

    def _add_custom(self,
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
                par = self._nodes.get(par)
                assert par is not None, f"Unknown parent '{par.name}'"
            inputs.append(par)
        # eventually build the node
        function = lambda: func(*[inp.value for inp in inputs])
        return self._check_node_and_append(func=function, name=name, hidden=hidden, parents=set(inputs))

    def _check_node_and_append(self,
                               func: Callable[..., np.ndarray],
                               name: Optional[str],
                               parents: Set[Node],
                               hidden: Optional[bool]) -> Node:
        # assign default name if no name is passed, otherwise check for name conflicts with existing variables
        if name is None:
            self._counter += 1
            name = f'v{self._counter}'
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
        SIZE.set(num)
        for node in nx.topological_sort(self._graph):
            data[node] = self._nodes[node].sample()
        for node in self._nodes.values():
            node.clear()
        SIZE.clear()
        return data if hidden else data[[n.name for n in self.visible]]

    def visualize(self):
        g = self._graph.copy()
        for layer, nodes in enumerate(nx.topological_generations(g)):
            for node in nodes:
                g.nodes[node]['layer'] = layer
        pos = nx.multipartite_layout(g, subset_key='layer')
        color = ['#3466AA' if n.visible else '#C0C0C0' for n in self.nodes]
        nx.draw(g, pos=pos, node_color=color, with_labels=True, arrows=True)
        plt.gca().collections[0].set_edgecolor("#000000")
        plt.show()
