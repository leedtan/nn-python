import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import math
#import nn_python as nn_p

def kfoldvalidation(X, Y, net, k=4, graph=False, ** kwargs):
    if len(X) != len(Y):
        print ("X and Y different lengths in kfoldval")
        sys.exit(0)
    perm = np.random.permutation(len(Y))
    num_trn = len(X)
    y_predict = np.empty_like(Y)
    for k_idx in range(k):
        val_idx = perm[math.floor(k_idx * 1.0 / k * num_trn):
                       math.floor((k_idx + 1) * 1.0 / k * num_trn)]
        if k_idx == k - 1:
            val_idx = perm[math.floor(k_idx * 1.0 / k * num_trn):]
        y_predict[val_idx] = predict_fold(X, Y, net, perm, val_idx, ** kwargs)
    sum_error = np.sum(np.abs(y_predict - Y))
    if graph:
        plt.plot(Y, label='Validation Values')
        plt.plot(y_predict, label='Predicted Values')
        plt.legend(bbox_to_anchor=(0., 0.01, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
        plt.title('k-fold Validating. Mean Accuracy = ' +
                  str(sum_error / num_trn))
        plt.show()
    return [sum_error / num_trn]

def predict_fold(X, Y, net, perm, val_idx, **kwargs):
    train_net = copy.deepcopy(net)
    trn_idx = [i for i in perm if i not in val_idx]
    train_net.train(X[trn_idx], Y[trn_idx], **kwargs)
    return train_net.predict(X[val_idx])
