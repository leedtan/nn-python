import numpy as np

# names of batch training types:
SOLO = -1
GROUP = 0

STOP_TRAIN = -1
cat = np.concatenate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -50, 50)))


def sigmoid_p(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_p(x):
    return 1.0 - x ** 2


def purelin(x):
    return x


#this might be wrong
def purelin_p(x):
    return -1


trans_fcns = {
              'sigmoid': [sigmoid, sigmoid_p],
              'tanh': [tanh, tanh_p],
              'purelin': [purelin, purelin_p]
              }


def mse(Y, a):
    return np.sum(np.square(Y - a))


def mse_p(Y, a):
    return (Y - a)


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


def print_y(y_predict, Y, dec=2):
    if y_predict.ndim > 1:
        y_predict = y_predict.flatten()
    print(np.around(y_predict, dec))
