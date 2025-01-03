# neural_network
Neural network (MLP) implemented from scratch in C++17
======================================================
Achieved accuracy of 89.9% on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

Forward pass and backpropagation implemented entirely in matrix form.
Some other implemented features include:
  - Data normalization
  - He/PyTorch Weight initialization
  - Adam stochastic optimizer

Building the implementation can be done via
  $ mkdir build && cd build && cmake -S ..
  $ make && ./neural-net

Afterwards, you can evaluate the accuracy on the Fashion-MNIST dataset using the provided evaluator like so:
  $ python3 ../evaluator/evaluate.py test_predictions.csv ../data/fashion_mnist_test_labels.csv 
