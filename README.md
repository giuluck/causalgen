## CAUSALGEN: A Causal-based Utility for Data Generation

_Causalgen_ is a utility that allows to generate data which is based on a certain causal graph structure.

### **1. Build A New Generator**

In order to generate your random data, you first need to create a Generator instance as:

```python
from causalgen import Generator

dg = Generator(seed=42)
```

The seed parameter controls the random number operations of the generator, it can be:
* an integer, in which case a new numpy random number generator (```rng```) is created
* an already existing ```rng```, in which case this ```rng``` is used within the generator
* None, in which case the ```np.random``` module is used instead

### **2. Add Source Variables**

Once you have your generator, you can add your random variables. 
Generator objects already provide some random distributions, which you can use to build a new variable.
When a new variable is added, the generator automatically appends it inside its data structure and returns it.
Moreover, each variable must have a unique name within the graph.
```python
A = dg.normal(mu=0, sigma=1, name='a')
```

> _Note: if no name is passed, one is automatically attached to a new node_

In case you want to use a custom random variable which is not provided, you can do that with the ```custom``` method.
This method works as all the other distribution methods, but you can pass a custom function to it.
```python
B = dg.custom(lambda s: dg.random.weibull(a=1, size=s), name='b')
```

> _Note: the function must match the signature ```f(size: int) -> np.ndarray```, where ```size``` represent the length of the output vector_

Additionally, you can create hidden variables using the ```hidden``` parameter.
Nodes marked as hidden should represent all those variables that cannot be measured in a real-world scenario.
These nodes will be displayed with a different color, and optionally removed from the generated dataframe.
```python
C = dg.integers(low=0, high=10, endpoint=False, hidden=True, name='c')
```

### **3. Add Child Variables**

Once you have build your source variables, you can add new descendant variables using the ```descendant``` method.
This method works similarly to the previous ones, but you can pass the input function in four different ways:
1. you can pass a custom function whose input parameters match the name of nodes already in the generator, in which case the generator automatically retrieves the parent nodes and build a new child node
```python
dg.descendant(lambda a, b, c: np.sin(a) + np.cos(b) - 3 * c, name='d1')
```

2. you can pass a function with a given number of parameters and then use the 'parents' argument to pass an (ordered) list of node instances that are already in the generator and whose values will be used as inputs
```python
dg.descendant(lambda x, y, z: np.sin(x) + np.cos(y) - 3 * z, parents=[A, B, C], name='d2')
```

3. you can use the same syntax as in 2. but passing the parents as strings instead of node objects, in which case the generator automatically retrieves the parent nodes and build a new child node
```python
dg.descendant(lambda x, y, z: np.sin(x) + np.cos(y) - 3 * z, parents=['a', 'b', 'c'], name='d3')
```

4. you can perform operations directly on nodes, as they support a wide number of numpy-based arithmetical and logical operations which are eventually transformed into a new node by the generator
```python
D4 = dg.descendant(np.sin(A) + np.cos(B) - 3 * C, name='d4')
```

### **4. Introduce Noise**

Another important aspect of data generation is the presence of noise.
When adding descendant nodes, you can introduce noise as part of the node's function:
1. you can use the 'noise' parameter in the 'descendant' method in order to add additive gaussian noise of the given amount
```python
dg.descendant(lambda d1: d1, noise=0.1, name='e1')
```
> _Note: this is usually enough, but if your noise should be either non-gaussian or non-additive you will have to leverage one of the other two methods_

2. if you create the node by passing a user-defined function, you can add noise using the internal random number generator which is stored in the ```random``` field;
please be careful to use the internal rng instead of other random number generators (or the ```np.random``` package) in order to get reproducible results
```python
dg.descendant(lambda d2: d2 + 0.1 * dg.random.normal(), name='e2')
```

3. if you create the node via explicit variable's operations, you can simply create a new source within the operation itself;
this is equivalent to adding a new node and then use it within the equation, hence the resulting object will appear both in visualization and in the generated dataframe
```python
dg.descendant(D4 + 0.1 * dg.noise(), name='e4')
```

### **5. Generate Data And Visualize Graph**

Now that you have built the causal graph, you can sample some instances using the ```generate``` method.
By default, the method returns a dataframe with the given number of instances and with visible variables only:
```python
df = dg.generate(num=10)
```

If you want to have access to hidden variables as well, you just need to use the ```hidden``` parameter.
```python
df = dg.generate(num=10, hidden=True)
```

Finally, you can visualize the casual graph structure using the ```visualize``` method.
```python
dg.visualize()
```