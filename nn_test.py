from nn_python import *  # noqa

def test_xor(verb=0, re_init=10, netstruc='ff'):
    tNN = nnp([2, 100, 1], reg=10 ** -10, netstruc=netstruc)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([np.array([x[0] ^ x[1]]) for x in X])
    tNN.train(X, y, epochs=1000, verb=verb, batch_type=2,
              re_init=re_init, re_init_d=100)
    y_predict = tNN.predict(X)
    if verb > -1:
        print (np.around(y_predict, 2))
    if y.ndim == 1:
        y_predict = y_predict.flatten()
    if verb > -1:
        print (np.mean(np.square(y_predict - y)))
    return np.mean(np.square(y_predict - y))
np.random.seed(104)
test_xor(verb = 0)