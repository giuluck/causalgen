import inspect
from typing import Callable, Optional, Dict, Union, List, Set, Tuple, Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from causalgen.variables import Variable, Node, Intermediate


class Generator:
    """
    A Causal Graph Based Data Generator.

    It contains information about the random variables and their causal structure, and allows to perform common
    operations such as variable addition, seed-controlled data generation, and visualization of the causal graph.
    """

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
        """
        :param seed:
            Either an integer seed, a numpy random number generator, or None.
            If a rng instance is passed, this is used as internal rng for random operations.
            If an integer is passed, a new internal rng is built using that seed.
            If None is passed, the 'np.random' module is used instead.
        """
        self._rng: np.random.Generator = Generator._get_rng(seed)
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, Node] = {}
        self._counters: Dict[str, int] = {}
        # internal counters for automatic node name assignment
        self._size: Optional[int] = None
        # internal 'size' variable which is used to generate data from source nodes

    @property
    def random(self) -> np.random.Generator:
        """The internal random number generator, or 'np.random' if None was passed as seed."""
        return self._rng

    @property
    def graph(self) -> nx.DiGraph:
        """A networkx directed graph instance which represents the underlying causal graph."""
        return self._graph.copy()

    @property
    def nodes(self) -> List[Node]:
        """The list of nodes in the causal graph."""
        return [v for v in self._nodes.values()]

    @property
    def hidden(self) -> List[Node]:
        """The list of hidden node variables in the causal graph."""
        return [v for v in self._nodes.values() if v.hidden]

    @property
    def visible(self) -> List[Node]:
        """The list of visible node variables in the causal graph."""
        return [v for v in self._nodes.values() if v.visible]

    @property
    def sources(self) -> List[Node]:
        """The list of source node variables in the causal graph."""
        return [v for v in self._nodes.values() if v.source]

    def reset_seed(self, seed: Union[None, int, np.random.Generator]) -> Any:
        """Resets the internal random number generator.

        :param seed:
            Either an integer seed, a numpy random number generator, or None.
            If a rng instance is passed, this is used as internal rng for random operations.
            If an integer is passed, a new internal rng is built using that seed.
            If None is passed, the 'np.random' module is used instead.

        :return:
            The generator itself.
        """
        self._rng = Generator._get_rng(seed)
        return self

    def constant(self, value: float = 0.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node with constant value.

        :param value:
            The constant value to be set.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'noise_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: np.array([value] * self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='const')

    def noise(self, amount: float = 1.0, hidden: bool = True, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a zero-mean normal distribution.

        :param amount:
            The amount of noise.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'noise_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.normal(loc=0.0, scale=amount, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='noise')

    def uniform(self, low: float = 0.0, high: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a uniform distribution.

        :param low:
            The lower bound of the uniform distribution.

        :param high:
            The upper bound of the uniform distribution.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'uni_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.uniform(low=low, high=high, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='uni')

    def normal(self, mu: float = 0.0, sigma: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a normal distribution.

        :param mu:
            The mean of the normal distribution.

        :param sigma:
            The variance of the normal distribution.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'norm_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.normal(loc=mu, scale=sigma, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='norm')

    def lognormal(self, mu: float = 0.0, sigma: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a lognormal distribution.

        :param mu:
            The mean of the lognormal distribution.

        :param sigma:
            The variance of the lognormal distribution.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'lnorm_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.lognormal(mean=mu, sigma=sigma, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='lnorm')

    def exponential(self, scale: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from an exponential distribution.

        :param scale:
            The scale of the exponential distribution.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'exp_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.exponential(scale=scale, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='exp')

    def poisson(self, lam: float = 1.0, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a poisson distribution.

        :param lam:
            The lambda of the poisson distribution.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'pois_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.poisson(lam=lam, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='pois')

    def geometric(self, p: float = 0.5, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a geometric distribution.

        :param p:
            The probability of the geometric distribution.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'geom_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.geometric(p=p, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='geom')

    def binomial(self, p: float = 0.5, hidden: bool = False, name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a binomial distribution.

        :param p:
            The probability of the binomial distribution.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'bin_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.binomial(n=1, p=p, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='bin')

    def integers(self,
                 low: int = 0,
                 high: int = 1,
                 endpoint: bool = True,
                 hidden: bool = False,
                 name: Optional[str] = None) -> Node:
        """Creates a new source node which samples integer values.

        :param low:
            The lower integer value.

        :param high:
            The upper integer value.

        :param endpoint:
            Whether to include the upper integer value ([low, high]) or not ([low, high]).

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'int_{i}' is assigned.

        :return:
            The created source node.
        """
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
        """Creates a new source node which samples from the given list of categorical values.

        :param categories:
            The list of categorical values to be sampled.

        :param replace:
            Whether to sample with replacement or not.

        :param p:
            The probability of each possible value to be sampled. If None, each value has probability p = 1 / n.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'cat_{i}' is assigned.

        :return:
            The created source node.
        """
        func = lambda: self._rng.choice(a=categories, replace=replace, p=p, size=self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist='cat')

    def custom(self,
               distribution: Callable[[int], np.ndarray],
               hidden: bool = False,
               name: Optional[str] = None) -> Node:
        """Creates a new source node which samples from a custom distribution.

        :param distribution:
            A function f(size: int) -> np.ndarray which returns a vector of length size.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'var_{i}' is assigned.

        :return:
            The created source node.
        """
        # check that the distribution signature matches the expected one (i.e., f(int) -> array)
        signature = inspect.signature(distribution).parameters.keys()
        assert len(signature) == 1, f"Custom distributions should match signature f(size: int) -> np.ndarray"
        func = lambda: distribution(self._size)
        return self._check_node_and_append(func=func, hidden=hidden, name=name, parents=set(), dist=None)

    def descendant(self,
                   function: Union[Variable, Callable[..., np.ndarray]],
                   parents: Optional[List[Union[str, Node]]] = None,
                   noise: Optional[float] = None,
                   name: Optional[str] = None,
                   hidden: bool = False) -> Node:
        """Creates a new descendant node.

        :param function:
            The function which defines the node value.
            If a Callable object is passed, the number of its arguments must match the number of given parents
            (otherwise, it is possible to set parents=None, in which case the list of input arguments will be
            automatically retrieved from the function signature, and their names will be passed as parents).
            Otherwise, if a Variable object is passed, a new node is built from its value.

        :param parents:
            The (ordered) list of node parents, which must be already present in the generator.
            Parents can be passed either as objects or as strings (i.e., using their node names).
            If the given function is not a Callable object but a Variable object, which results from a series of
            Variable operations, this parameter should not be used (i.e., its value must be None).

        :param noise:
            The amount of (additive) gaussian noise, or None for no noise.

        :param hidden:
            Whether the source node should be hidden in the causal graph or not.

        :param name:
            The name of the node. If None, name 'var_{i}' is assigned.

        :return:
            The created descendant node.
        """
        if isinstance(function, Variable):
            assert parents is None, f"Derived variables must have no parents, please use 'parents=None'"
            return self._add_var_descendant(var=function, name=name, noise=noise, hidden=hidden)
        elif isinstance(function, Callable):
            return self._add_func_descendant(func=function, name=name, noise=noise, hidden=hidden, parents=parents)
        else:
            raise TypeError(f"Unsupported function type '{type(function).__name__}'")

    def _add_var_descendant(self, var: Variable, noise: Optional[float], name: Optional[str], hidden: bool) -> Node:
        # if a node is passed simply use it as parents, otherwise retrieve all the parents from intermediate operations
        if isinstance(var, Node):
            # otherwise, the new node is simply an identity function from the previous one
            parents = {var}
        elif isinstance(var, Intermediate):
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
        else:
            raise TypeError(f"Unknown variable type '{type(var).__name__}'")
        # the function is computed as an identity function of the given variable value, plus noise if necessary
        if noise is None:
            function = lambda: var.value
        else:
            function = lambda: var.value + self._rng.normal(scale=noise, size=self._size)
        # eventually build and append the node
        return self._check_node_and_append(func=function, parents=parents, hidden=hidden, name=name)

    def _add_func_descendant(self,
                             func: Callable[..., np.ndarray],
                             noise: Optional[float],
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
        # the function is computed by passing the parents values as inputs, plus noise if necessary
        if noise is None:
            function = lambda: func(*[inp.value for inp in inputs])
        else:
            function = lambda: func(*[inp.value for inp in inputs]) + self._rng.normal(scale=noise, size=self._size)
        # eventually build and append the node
        return self._check_node_and_append(func=function, name=name, hidden=hidden, parents=set(inputs), dist=None)

    def _check_node_and_append(self,
                               func: Callable,
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
        """Generates a new dataframe using the random variables in the graph.

        :param num:
            The number of instances to generate.

        :param hidden:
            Whether to include hidden variables in the output dataframe or not.

        :return:
            A pandas dataframe instance with one column for each of the variables.
        """
        data = pd.DataFrame()
        # check that no unexpected behaviour is going on and then assign the size internal value
        assert self._size is None, "Unexpected behaviour, internal variable '_size' should be None but it is not"
        self._size = num
        # use topological sort to sample each node from the sources to the descendants
        for node in nx.topological_sort(self._graph):
            data[node] = self._nodes[node].sample()
        # then clear all the nodes in order to be consistent with the Variable's API
        for node in self._nodes.values():
            node.clear()
        # and clear the size internal value as well
        self._size = None
        # eventually return all the (ordered) data or the subset of visible data according to the hidden parameter
        return data[[n.name for n in (self.nodes if hidden else self.visible)]]

    def visualize(self,
                  pos: Union[str, List[Iterable[Optional[str]]], Dict[str, Any]] = 'lp',
                  fig_size: Tuple[int, int] = (16, 9),
                  padding: Union[float, Tuple[float, float]] = 0.1,
                  font_size: float = 16,
                  arrow_pad: float = 10000,
                  arrow_size: float = 50,
                  edge_width: float = 1):
        """Visualizes the underlying casual graph.

        :param pos:
            Defines the position of the nodes in the visualization.
            If a dictionary is passed, it should represent the absolute position of each node indexed by name.
            If a list is passed, it should contain one sub-list per horizontal layer, and the ordered names of each
            node that appear in that layer, or None for an empty placeholder.
            If the string 'sp' is passed, nodes are arranged in layers using the shortest path algorithm from sources.
            If the string 'lp' is passed, nodes are arranged in layers using the longest path algorithm from sources.

        :param fig_size:
            The figsize parameter passed to plt.figure().

        :param padding:
            The amount of padding outside the whole graph, either a float or a tuple (horizontal, vertical).

        :param font_size:
            The size of the font.

        :param arrow_pad:
            The padding of each arrow.

        :param arrow_size:
            The size of each arrow.

        :param edge_width:
            The width of each edge (and node borders).
        """
        # if the position is a string, use the respective strategy to layer nodes horizontally into a list of lists
        if isinstance(pos, str):
            # noinspection PyTypeChecker
            sources = [node for node, degree in self._graph.in_degree if degree == 0]
            if pos == 'sp':
                # use breadth first search for shortest paths
                pos = [sorted(nodes) for nodes in nx.bfs_layers(self._graph, sources=sources)]
            elif pos == 'lp':
                # use floyd warshall algorithm to search for longest paths
                #  - get the indices of the sources
                #  - get the negative shortest path matrix and select only the paths from the sources
                #  - get the negative minimum value for each node in the graph and negate it to get the layer
                gcopy = self._graph.copy()
                nx.set_edge_attributes(gcopy, values=-1, name='weight')
                sources = [i for i, node in enumerate(gcopy.nodes) if node in sources]
                lp = -nx.floyd_warshall_numpy(gcopy)[sources].min(axis=0)
                layers = {}
                for i, node in enumerate(gcopy.nodes):
                    layer = lp[i]
                    if layer not in layers:
                        layers[layer] = [node]
                    else:
                        layers[layer].append(node)
                pos = [sorted(nodes) for _, nodes in sorted(layers.items(), key=lambda v: v[0])]
            else:
                raise ValueError(f"Unknown 'pos' value '{pos}'")
        # if the position info is a list, build positions by vertically spacing nodes in the same layer
        if isinstance(pos, list):
            layers = pos
            pos = {}
            for x, nodes in enumerate(layers):
                y = np.linspace(1, 0, len(nodes) + 2)[1:-1]
                pos.update({n: (x, y) for n, y in zip(nodes, y) if n is not None})
        # eventually draw the graph (draw boxed labels rather than nodes, and draw edges and arrows separately)
        fig = plt.figure(figsize=fig_size, tight_layout=True)
        for nodelist, color in [(self.visible, '#73B9EE'), (self.hidden, '#C0C0C0')]:
            nx.draw_networkx_labels(
                self._graph.subgraph(nodes=[node.name for node in nodelist]),
                pos=pos,
                font_size=font_size,
                bbox=dict(facecolor=color, edgecolor='black', boxstyle=f'round,pad=1.0'),
                ax=fig.gca()
            )
        nx.draw_networkx_edges(self._graph, pos=pos, node_size=arrow_pad, width=0, arrowsize=arrow_size, ax=fig.gca())
        nx.draw_networkx_edges(self._graph, pos=pos, node_size=0, width=edge_width, arrows=False, ax=fig.gca())
        # set axis boundaries and show figure
        x_pad, y_pad = padding if isinstance(padding, tuple) else (padding, padding)
        x_pos = [x for x, _ in pos.values()]
        y_pos = [y for _, y in pos.values()]
        x_min, x_max = min(x_pos), max(x_pos)
        y_min, y_max = min(y_pos), max(y_pos)
        x_pad *= x_max - x_min
        y_pad *= y_max - y_min
        fig.gca().set_xlim(x_min - x_pad, x_max + x_pad)
        fig.gca().set_ylim(y_min - y_pad, y_max + y_pad)
        fig.show()
