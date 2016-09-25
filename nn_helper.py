import numpy as np

# names of batch training types:
solo = -1
group = 0

cat = np.concatenate

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))


def sigmoid_p(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def mse(y, a):
    return np.sum(np.square(y - a))


def mse_p(y, a):
    return (y - a)


def copy_weights(weights):
    return [[np.array([z for z in w]) if (len(w) != 1 or w[0] != 0)
             else [0] for w in x] for x in weights]


def zero_weights(weights):
    return [[np.array([np.zeros_like(z) for z in w]) if
             (len(w) != 1 or w[0] != 0) else [0] for w in x] for x in weights]


def apply_norm(X, Xstd, Xoffset):
    return (X - Xoffset)/Xstd

def normalize(vect):
    offset = np.mean(vect)
    vect = vect - offset
    stdev = np.std(vect)
    vect = vect / stdev
    return [vect, stdev, offset]


def denormalize(vect, stdev, offset):
    return (vect * stdev) + offset
