# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Long Term TODO : 
- transformation

## Mid Term TODO : 
- Refactor
- De-label
- generics

## TODO : Get Mnist back at work
- Fix mnist
- Have a similar test for mnist like uci income
- new ver

## [0.0.1] - 2022-07-16
- Major refactoring
    - Major reasons 
      - Not relying on type checking, using a `validate` function to wrap around
      - the core representation not being convenient to use, but concise on definition
          - there could be a frontend and grammar sugar to solve this later
    - Concepts
      - The TensorHub representation
      - L: label, DF: dimension family, D: dimensions, B: BaseVariable, T: Tensor, H: Hub
    - Rules
      - L <- L([label])
      - DF <- DF(positioned, L, n_members, Optional[len])
      - D <- [DF]
      - B <- Onehot|Bounded|Gamma|...
      - T <- T(D, B, L)
      - H <- H([T])
      - Special rules:
        - H cannot be empty
      - Notes:
        - note that D can be empty to represent a scalar
        - note that L can be empty for no labels
    - Derivation
      - the reason for the tensorhub representation is in T(D, X, L), if X is a tensor,
        then X can be expanded to T, if X is a group of variables, then it can be expanded to a group of tensors. 
      - therefore tensor can just hole base variable
      - for complex group structures, it can be flattened by multiple labels,
        the flattened structure is much more convenient to be dealt with than trees
- Added sugar
  - usage details in `common_variables.py`