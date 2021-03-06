# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## TODO : Get Mnist back at work
- Transformations:
    - directional
    - 2 tier
        - Base Transformation (between vars)
            - Anytype to anytype
        - Tensor Transformation (reshape, squeeze)

## [0.0.3] - 2022-07-22
- [BaseVariable] fixed a bug on ord labels
- [Sugar] added fast scalar

## [0.0.2] - 2022-07-19
- made tensorhub validation compulsory
    - post_init will ensure Tensorhub checked

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
  - built around `Dim`