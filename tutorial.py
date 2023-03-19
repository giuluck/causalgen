import pandas as pd

from datagen import Generator

pd.set_option('precision', 2)

if __name__ == '__main__':
    dg = Generator(seed=42)

    # create sources
    A = dg.normal(mu=0, sigma=1, name='a')
    B = dg.custom(lambda s: dg.random.uniform(low=0, high=1, size=s), name='b')
    C = dg.integers(low=0, high=10, endpoint=False, hidden=True, name='c')

    # create descendants
    dg.descendant(lambda a, b, c: a * b + c, name='d1')
    dg.descendant(lambda x, y, z: x * y + z, parents=[A, B, C], name='d2')
    dg.descendant(lambda x, y, z: x * y + z, parents=['a', 'b', 'c'], name='d3')
    D4 = dg.descendant(A * B + C, name='d4')

    # introduce noise
    dg.descendant(lambda d1: d1 + dg.random.normal(scale=0.1), name='e1')
    dg.descendant(D4 + dg.noise(amount=0.1), name='e4')

    print('Generated Dataset Without Hidden Nodes')
    print(dg.generate(num=10))

    dg.reset_seed(42)
    print('\n\nGenerated Dataset With Hidden Nodes')
    print(dg.generate(num=10, hidden=True))

    dg.visualize(node_size=25, arrow_size=12, edge_width=1.5)
