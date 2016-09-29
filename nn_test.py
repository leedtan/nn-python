from nn_python import *  # noqa

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
for _ in range(100):
    test_xor()