import numpy as np

import config


def layer_sizes(hidden=None):
    """Sizes of the layers. hidden=0 gives a network without a hidden layer."""
    hidden = config.N_HIDDEN if hidden is None else hidden
    if hidden == 0:
        return [config.N_INPUTS, config.N_OUTPUTS]
    return [config.N_INPUTS, hidden, config.N_OUTPUTS]


def chromosome_length(sizes):
    """Number of weights and biases in the network."""
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def decode(chromosomes, sizes):
    """Cut the flat chromosomes into weight matrices and bias vectors.
    chromosomes has shape (N, length), one row per car."""
    layers = []
    start = 0
    n = len(chromosomes)
    for a, b in zip(sizes[:-1], sizes[1:]):
        w = chromosomes[:, start:start + a * b].reshape(n, a, b)
        start += a * b
        bias = chromosomes[:, start:start + b]
        start += b
        layers.append((w, bias))
    return layers


def forward(layers, x):
    """Forward pass for all the cars at once.
    x has shape (N, inputs). Returns the activations of every layer."""
    activations = [x]
    for w, bias in layers:
        # for every car n:  x[n] @ w[n] + bias[n]
        x = np.tanh(np.einsum("ni,nio->no", x, w) + bias)
        activations.append(x)
    return activations


def drive(layers, x):
    """Returns the steering and the throttle for every car."""
    out = forward(layers, x)[-1]
    return out[:, 0], out[:, 1]


def sizes_from_length(length):
    """Find the layer sizes of a saved chromosome from its length."""
    if length == chromosome_length(layer_sizes(0)):
        return layer_sizes(0)
    for hidden in range(1, 200):
        if chromosome_length(layer_sizes(hidden)) == length:
            return layer_sizes(hidden)
    raise ValueError("unknown chromosome length %d" % length)
