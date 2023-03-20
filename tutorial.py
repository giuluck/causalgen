import numpy as np
import pandas as pd

from datagen import Generator

pd.set_option('precision', 2)

if __name__ == '__main__':
    # as a first step, we need to create a Generator instance
    # the seed parameter controls the random number operations of the generator, it can be:
    #   - an integer, in which case a new numpy random number generator (rng) is created
    #   - an already existing rng, in which case this rng is used within the generator
    #   - None, in which case the 'np.random' module is used instead
    dg = Generator(seed=42)

    # once you have your generator, you can add your random variables
    # generator objects already provide some random distributions, which you can use to build a new variable
    # each variable has a name (if no name is passed, one is automatically attached to it)
    # when a new variable is added, the generator automatically appends it inside its data structure and returns it
    A = dg.normal(mu=0, sigma=1, name='a')

    # in case you want to use a custom random variable which is not provided, you can do that with the 'custom' method
    # this method works as all the other distribution methods, but you can pass a custom function to it
    # the function must match the signature f(size: int) -> np.ndarray, where size represent the length of the output
    B = dg.custom(lambda s: dg.random.weibull(a=1, size=s), name='b')

    # additionally, you can create hidden variables using the 'hidden' parameter
    # nodes marked as hidden should represent all those variables that cannot be measured in a real-world scenario
    # these nodes will be displayed with a different color, and optionally from the generated dataframe
    C = dg.integers(low=0, high=10, endpoint=False, hidden=True, name='c')

    # once you have build your source variables, you can add new descendant variables using the 'descendant' method
    # this method works similarly to the previous ones, but you can pass the input function in four different ways:
    #    1. you can pass a custom function whose input parameters match the name of nodes already in the generator, in
    #       which case the generator automatically retrieves the parent nodes and build a new child node
    #    2. you can pass a function with a given number of parameters and then use the 'parents' argument to pass an
    #       (ordered) list of node instances that are already in the generator whose values will be used as inputs
    #    3. you can use the same syntax as in 2. but passing the parents as strings instead of node objects, in which
    #       case the generator automatically retrieves the parent nodes and build a new child node
    #    4. you can perform operations directly on nodes, as they support a wide number of numpy-based arithmetical and
    #       logical operations which are eventually transformed into a new node by the generator
    dg.descendant(lambda a, b, c: np.sin(a) + np.cos(b) - 3 * c, name='d1')
    dg.descendant(lambda x, y, z: np.sin(x) + np.cos(y) - 3 * z, parents=[A, B, C], name='d2')
    dg.descendant(lambda x, y, z: np.sin(x) + np.cos(y) - 3 * z, parents=['a', 'b', 'c'], name='d3')
    D4 = dg.descendant(np.sin(A) + np.cos(B) - 3 * C, name='d4')

    # finally, another important aspect of data generation is the presence of noise
    # when adding descendant nodes, you can introduce noise as part of the node's function:
    #   - if you create the node by passing a user-defined function, you can add noise by leveraging the internal
    #     random number generator which is stored in the 'random' field; please be careful to use the internal rng
    #     instead of other random number generators (or the np.random package) in order to get reproducible results
    #   - if you create the node via explicit variable's operations, you can simply create a new source within the
    #     operation itself; in this case, the 'noise' method is very useful to add noise, but differently from the
    #     previous methodology of introducing noise, here the resulting noise vector will be stored as a (hidden by
    #     default) variable in the generator instance -- see the 'noise_1' column in the generated dataframe
    dg.descendant(lambda d1: d1 + 0.1 * dg.random.normal(), name='e1')
    dg.descendant(D4 + 0.1 * dg.noise(), name='e4')

    # now that you have built the causal graph, you can sample some instances using the 'generate' method
    # by default, the method returns a dataframe with the given number of instances and with visible variables only
    print('Generated Dataset Without Hidden Nodes')
    print(dg.generate(num=10))

    # if you want to have access to hidden variables as well, you just need to use the 'hidden' parameter
    # also, the generator instance has a 'reset_seed' method which allow to reset its internal rng; in this case, we
    # can use this method in order to generate the same instances as for the previous generation
    dg.reset_seed(42)
    print('\n\nGenerated Dataset With Hidden Nodes')
    print(dg.generate(num=10, hidden=True))

    # eventually, you can visualize the casual graph structure using the 'visualize' method
    dg.visualize(fig_size=(10, 6), node_size=25, arrow_size=12, edge_width=1.5)
