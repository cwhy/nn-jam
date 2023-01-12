# Functions, Pipelines, Processes

From functions in functions.py to pipeline to processes,
the more complex a function will be.

- Function handles a certain numerical calculation.
- FixedPipelines are functions with weights(as parameters), but only one input and output
- FixedProcesses are functions with weights(as parameters) and multiple input/outputs
- Non-fixed version of the above are ones with an extra `rng` argument as random seed

For weights and inputs with multiple arrays, it is in a format of ArrayTreeMapping.
A mapping from string to an array or an ArrayTreeMapping.



```
functions.py Functions: [Arrays]->Array


           ┌─────────────────────────────────────────────────────┐
           │                                                     │
Pipelines: │ ┌──────────────────────────────────────┐            │
           │ │    weights             x             │    rng     │
           │ │                                      │            │
           │ │ ArrayTreeMapping      Array          │    RNG     │  -> Array
           │ └──────────────────────────────────────┘            │
           │                FixedPipeline                        │
           │                                                     │
           └─────────────────────────────────────────────────────┘

          ┌─────────────────────────────────────────────────────┐
          │                                                     │
Process:  │ ┌──────────────────────────────────────┐            │
          │ │    weights            inputs         │    rng     │
          │ │                                      │            │
          │ │ ArrayTreeMapping  ArrayTreeMapping   │    RNG     │  -> ArrayTreeMapping
          │ └──────────────────────────────────────┘            │
          │                FixedPipeline                        │
          │                                                     │
          └─────────────────────────────────────────────────────┘
```

## Components

A component is a standard unit of `jax_make`, it is the output of the `make` function.
And it is used to be trained and making inference.

It consists of `weight_params`, a tree of weight configs, that is used for weight initialization,
and `processes`, a dict of `ProcessPorts` to `Process`, which is used as functions that is passed into jax.


```
        Component:
                         from_pipeline, from_process, from_fixed_pipeline
        ┌┬──────────────────────────────────────────────────────────┐   │
        ││                                                          │   │
        ││                     ProcessPort┌─────────────────┐       │◄──┘
        ││                          ┌─────┤  Process        │       │
        ││     processes: ──────────┤     └─────────────────┘       │
        ││                          │                               │
        ││     weight_params:       │      ┌──────────────────┐     │
        ││     ┌─────────────────┐  └┬─────┤   Process        │     │
        ││     │ArrayParamMapping│   │     └──────────────────┘     │
        ││     └─────────────────┘   │                              │
        ││                           │       ┌──────────────────┐   │
        ││                           └───────┤    Process       │   │
        ││     is_fixed: bool                └──────────────────┘   │
        ││                                                          │
        └┴──────┬───────────────────────────────────────────────────┘
                │
                └───► pipeline, fixed_pipeline (If there is only one process
                                                and it is a pipeline)
```