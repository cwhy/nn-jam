# Jax-make
A jax mini-framework for modularized jax experience in writing neural-networks
This is written for people who are OK with frameworks, keep away from this if you are (some of those people)[https://news.ycombinator.com/item?id=28920095].


## The problem of pytorch/tf2.0+/most major framework
For a NN module, there are two stages of computation: the initialization, and the evaluation.
The speed for initialization does not usually matter, but evaluation time matters a lot.
Modularizing stuff will inherently introduce overheads,
 the approach of major current frameworks provides an OO encapsulation way for users to write modules in their way.
But that limits the expressiveness, for network beyond `Sequencial`, a lot of edge cases exist that this OO abstraction cannot handle.
If the developer does not know what is the magic hidden behind the OO framework, 
 they cannot fall back to use the expressiveness of the Python language without fearing to sacrifice the performance.


TF1 and jax just computes and does not include a module system.
Jax-make is based on jax and created a more flexible module system that has explicit initialization and evaluation steps.
It also utilized a lot of Python typing helpers for checking configurations.
The philosophy is that you will be writing your own mini-compilers for your NN and you can control the compiling stage with ease.


What about Haiku?
Since it strive to be a library, it needs to satisfy needs of a variety of use cases and there are a lot of configurations.
Jax-make is designed to be a mini-framework, 
 which includes some abstractions that will limit the way to do things.
In this way, I hope the framework to be small and keep opinionated,
 so that the code base will be small and easy to tinker around.


## Concepts
Now it's early stage, they all may be changed
### Params/Weights
Thanks to Jax's elegant way to handle Python tree structures seamlessly,
in jax-make, all parameters of the model are store in a single dict.

So you can have:
```bash
pprint(tree_map(lambda _x: _x.shape, weights))
```
```python
{'encoder': {'norm': {'a': (1,), 'b': (1,)},
             'tfe_layer_0': {'dropout': {},
                             'mha': {'kqv': {'b': (36,), 'w': (12, 36)},
                                     'out': {'b': (12,), 'w': (12, 12)}},
                             'mlp': {'layer_0': {'b': (24,), 'w': (12, 24)},
                                     'layer_1': {'b': (12,), 'w': (24, 12)}},
                             'norm1': {'a': (1,), 'b': (1,)},
                             'norm2': {'a': (1,), 'b': (1,)}},
             'tfe_layer_1': {'dropout': {},
                             'mha': {'kqv': {'b': (36,), 'w': (12, 36)},
                                     'out': {'b': (12,), 'w': (12, 12)}},
                             'mlp': {'layer_0': {'b': (24,), 'w': (12, 24)},
                                     'layer_1': {'b': (12,), 'w': (24, 12)}},
                             'norm1': {'a': (1,), 'b': (1,)},
                             'norm2': {'a': (1,), 'b': (1,)}},
             'tfe_layer_2': {'dropout': {},
                             'mha': {'kqv': {'b': (36,), 'w': (12, 36)},
                                     'out': {'b': (12,), 'w': (12, 12)}},
                             'mlp': {'layer_0': {'b': (24,), 'w': (12, 24)},
                                     'layer_1': {'b': (12,), 'w': (24, 12)}},
                             'norm1': {'a': (1,), 'b': (1,)},
                             'norm2': {'a': (1,), 'b': (1,)}},
             'tfe_layer_3': {'dropout': {},
                             'mha': {'kqv': {'b': (36,), 'w': (12, 36)},
                                     'out': {'b': (12,), 'w': (12, 12)}},
                             'mlp': {'layer_0': {'b': (24,), 'w': (12, 24)},
                                     'layer_1': {'b': (12,), 'w': (24, 12)}},
                             'norm1': {'a': (1,), 'b': (1,)},
                             'norm2': {'a': (1,), 'b': (1,)}}},
 'mask_embedding': {'dict': (1, 12)},
 'norm': {'a': (1,), 'b': (1,)},
 'out_embedding': {'dict': (5, 12)},
 'patching': {'mlp': {'layer_0': {'b': (12,), 'w': (16, 12)},
                      'layer_1': {'b': (12,), 'w': (12, 12)}}},
 'positional_encoding': {'encoding_dim_0': (12, 7),
                         'encoding_dim_1': (12, 7),
                         'encoding_dim_2': (12, 1)},
 'positional_encoding_y': {'encoding_dim_0': (12, 1)}}
```

The parameters can be updated individually or all-together.
All updates will not be in-place, as defined by Jax.

### WeightParams  # TODO: change it to WeightConfig ? 
The parameters are created after initialization, and the configuration of the parameters will be created before that.
The definition of weight configs is
```python
ArrayGen = Literal['kaiming', 'dropout', 'embedding']
class WeightParams(NamedTuple):
   # from in to out
   shape: Tuple[int, ...]
   init: Union[ArrayGen, int, float] = "kaiming"
   scale: float = 1
```
All the information for initializing an array/tensor will be stored here.
`WeightParams` will also form a `dict`, which is used to generate all the weights using `make_weights`:
```python
weights = make_weights(weight_params)
```
### process
Process is a function that does the actual computation, it will take three parameters:
- params: a dict of parameters/weights that the computation requires
- x: a dict of jax.numpy array as inputs that the computation requires
- rng: a rng key, since jax is serious about pureness and there's not much harm about boilerplate, I keep it explicit
And it will return a dict of jax.numpy array as outputs.

All processes will be passed though jax's jit,
 so they need to be pure.

### ProcessPorts
```python
class ProcessPorts(NamedTuple):
    inputs: FrozenSet[str]
    outputs: FrozenSet[str]
```
It will store the available inputs and outputs for a process.
Variable protocol will be included in the future for shape checks.

### Components
A component represent uninitialized network modules. 
It consists of two major parts: weight-params and processes.
All the processes will be stored in a dict of `{ProcessPorts:Process}`

### make
A `make` function will take configuration as a parameter to make a component. It takes two inputs:

- config 
  The make function will take configuration of the components
  I used Python's Protocol (PEP 544) to specify the configs,
  so that the configs will not be unnecessarily nested.

So the overall structure is:

```
Config ------> Components ------------------> Params/process
        make                weight_init/jit

```
Since `make` is explict here, you can `make` sub-components inside another `make` function,
without worrying about performance -- since everything will be run once and jit-ed by jax~


## Sugars
Sugars are not all bad, as long as we are aware of them and manage them well.

### Config-make class
For convenience and disambiguation of all the `make` functions for different components,
 we utilize Python's oop as sugar, so that the configuration class will also include a `make` classmethod.
So that the pattern will be like:
```python
dropout = Dropout.make(Dropout(0.8))

```

### pipeline

### sequential

### fixed_process/fixed_pipeline

### Other sugars
Since all sugars I have written here (for my own needs) are functions. 
You can write yours too.
