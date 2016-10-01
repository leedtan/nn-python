from nn_python import *  # noqa
import time
import matplotlib.pyplot as plt
def test_xor(verb=0, re_init=10, netstruc='ff'):
    tNN = nnp([2, 100, 1], reg=10 ** -10, netstruc=netstruc)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([np.array([x[0] ^ x[1]]) for x in X])
    tNN.train(X, Y, epochs=100, verb=verb, batch_type=GROUP,
              re_init=re_init, re_init_d=100)
    y_predict = tNN.predict(X)
    if verb > -1:
        print_y(y_predict, Y)
    return np.mean(np.square(y_predict - Y))
np.random.seed(1)
test_xor()
    
    
def test_xor3(verb=0, netstruc='ff'):
    tNN = nnp([3, 500, 1], reg= 10 ** -1)
    X = np.array([[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    Y = np.array([np.array([x[0] ^ x[1] ^ x[2]]) for x in X])
    if verb > -1:
        print(Y.flatten())
    tNN.train(X, Y, epochs=100, verb=verb, re_init=10, re_init_d=20)
    y_predict = tNN.predict(X)
    if verb > -1:
        print_y(y_predict, Y)
    return np.mean(np.square(y_predict - Y))
test_xor3()


#Does not yet work well.
def test_sine(verb=0, netstruc='ff'):
    tNN = nnp([1, 100, 1], reg= 10 ** -5, trans='tanh')
    X = np.array([np.array([x]) for x in np.arange(0, 2 * np.pi, .1)])
    Y = np.array([np.sin(x) + np.random.randn(x.shape[0]) * .001
                      for x in X])
    tNN.train(X, Y, epochs=100, verb=verb, re_init=3, re_init_d=10)
    y_predict = tNN.predict(X)
    if verb > 0:
        print(Y.flatten())
    if verb > -1:
        print_y(y_predict, Y)
    return np.mean(np.square(y_predict - Y))
if 1:
    sine_error = test_sine(verb = 0)
    print('Sine error is: ', sine_error)

