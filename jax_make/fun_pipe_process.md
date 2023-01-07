# Functions, Pipelines, Processes

## Functions

Function: [Arrays]->Array

```

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
