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
    - Not relying on type checking, using a `validate` function to wrap around
    - Concepts
      - L <- L([label])
      - DF <- DF(positioned, L, n_members, Optional[len])
      - D <- [DF]
      - B <- Onehot|Bounded|Gamma|...
      - T <- T(D, B)
      - G <- G([T|G], Optional[L])
      - Special rules:
        - when G have only one member, L is required
      - Notes:
        - note that D can be empty to represent a scalar