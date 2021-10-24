# NN-Jam
Just like jamming in music, I jam in code with NN.
Same as in music, this help me create cool stuffs, grok things better, and therapeutic at the same time.

It is open so that it is convenient to share part of it with my friends.
After the projects are matured, I will separate them out to be proper open source projects.

I am jamming in Python3.8+

## Common concepts
### variable protocol

It is a lossless definition of data structure of all obtainable numeric data.

### port

A port is a sender or a receiver that takes/outputs variable of a certain protocol.

### model:  

A model has several ports that can send or receive data. 
It also contains a process to receive data in some ports and send out in others.

The model aim to be performing good.

### audition:

An audition is to test the basic-level performance of a model.

## Current Projects
It consists of three projects currently, there are *.md docs for each:
### Variable protocols:  
This defines the protocols for variables, and transformation of variables.
Every other projects may refer to the protocols defined here.

This project is not here yet, but currently I met a major design obstacle
and it is in a minimal usable state.
After that when I have the necessity to bring it here to jam with other projects.

### Model Audition(old name Supervised Benchmarks)
This project is to provide good interfaces to test models with benchmarks.
It was called supervised benchmarks because my original design was targeted at supervised learning.
But it is hard to say that self-supervise learning is not unsupervised learning right?

### Jax-make
This is a mini-framework for data modeling using jax the delightful library.
It will be used to make models, test in audition.

## Future Projects
### Anynet
The project is the origin of all current projects,
 it depends on a moderate implementation level of all three of them.
The idea of this project is written in another repo.

### ...
and more.. it may end up being a GUI project or a DSL for data modeling, who knows?