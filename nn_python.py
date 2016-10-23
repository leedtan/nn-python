import sys
import copy
import numpy as np
import nn_helper as helper


class nnp(object):

    def __init__(self, layers, trans='sigmoid', perf_fcn='mse',
                 reg=0, netstruc='ff'):
        self.Yscale, self.Yoffset = [0, 0]
        self.best_perf = np.inf
        self.layers = layers
        self.reg = reg
        self.Xstd, self.Xoffset = [], []
        self.depth = len(layers)
        self._settrans(trans)
        self._setperf(perf_fcn)
        self._initLmap(netstruc)
        self._initweights()
        self._init_act_vals()
        self.deltas = np.copy(self.act_vals)

    def _init_act_vals(self):
        self.act_vals = []
        for idx in range(len(self.layers)):
            self.act_vals.append(np.empty([self.layers[idx]]))

    def _setperf(self, perf_fcn):
        if perf_fcn == 'mse':
            self.perf_fcn = helper.mse
            self.perf_fcn_p = helper.mse_p

    def _initweights(self, std=.01):
        self.weights = [[[0] for _ in range(len(self.Lmap))]
                        for _ in range(len(self.Lmap[0]))]
        for r_idx in range(len(self.weights)):
            for c_idx in range(len(self.weights[r_idx])):
                if self.Lmap[r_idx][c_idx]:
                    self.weights[r_idx][c_idx] = (
                        np.random.randn(self.layers[r_idx] + 1,
                                        self.layers[c_idx]) * std
                    )

    def _set_one_trans(self, trans, idx):
        if trans in helper.trans_fcns:
            self.trans[idx], self.trans_p[idx] = helper.trans_fcns[trans]
        else:
            print("Error: " + str(trans) + " not known in tran_fcns")
            sys.exit(0)

    def _set_trans_list(self, trans):
        if len(trans) < len(self.layers) - 1:
            print('wrong number of transfer functions listed')
            sys.exit(0)
        for idx in range(1, len(self.layers)):
            self._set_one_trans(trans, idx)

    def _settrans(self, trans):
        self.trans = [0] * (len(self.layers))
        self.trans_p = [0] * (len(self.layers))
        if isinstance(trans, str):
            for idx in range(1, len(self.layers)):
                self._set_one_trans(trans, idx)
        elif isinstance(trans, list):
            self._set_trans_list(trans)

    def _initLmap(self, netstruc):
        if isinstance(netstruc, str):
            if netstruc == 'ff':
                self.Lmap = np.eye(self.depth, k=1)

    def copy_structure(self):
        new_NN = copy.deepcopy(self)
        new_NN._initweights()
        return new_NN

    def _prop_fwd(self):
        for idx in range(1, len(self.Lmap[0])):
            self.act_vals[idx] = self.trans[idx](
                self._sum_layer(idx))

    def _sum_layer(self, layer_idx):
        return np.sum(np.dot(
            helper.cat([self.act_vals[prev], [1]]),
            self.weights[prev][layer_idx])
            for prev in range(len(self.Lmap))
            if self.Lmap[prev][layer_idx])

    def _update_all_batch_deltas(self):
        for row in range(len(self.weights)):
            for col in range(len(self.weights[0])):
                if self.Lmap[row][col]:
                    self.weights[row][col] += self.batch_deltas[row][col]

    def _run_batch(self, batch, X, Y):
        self.batch_deltas = helper.zero_weights(self.weights)
        for i in batch:
            self._train_one_sample(X[i], Y[i])
        self._update_all_batch_deltas()
        err = self._test_batch(X, Y)
        return err

    def _run_epoch(self, X, Y):
        err = 0
        perm = np.random.permutation(self.samples)
        for batch_idx in range(self.num_batches):
            batch = perm[self.batch_size * batch_idx:self.batch_size *
                         (batch_idx + 1)]
            if batch_idx == self.num_batches - 1:
                batch = perm[self.batch_size * batch_idx:]
            err += self._run_batch(batch, X, Y)
        return err

    def _prop_back_one_layer(self, idx):
        return np.sum(
            self.deltas[pt].dot(self.weights[idx][pt][:-1].T *
                                self.trans_p[idx](self.act_vals[idx])
                                )
            for pt in range(len(self.Lmap)) if self.Lmap[idx][pt])

    def _prop_back(self):
        for idx in range(len(self.deltas) - 2, 0, -1):
            self.deltas[idx] = self._prop_back_one_layer(idx)

    def _add_error_to_one_delta_edge(self, row, col):
        self.batch_deltas[row][col] += self.LR * (
            np.atleast_2d(helper.cat((self.act_vals[row], [1]))).T
            .dot(np.atleast_2d(self.deltas[col])) -
            self.reg * self.weights[row][col] *
            (col < len(self.weights[0]) - 1)) / self.layers[row]

    def _add_error_to_delta_edges(self):
        for row in range(len(self.weights)):
            for col in range(len(self.weights[0])):
                if self.Lmap[row][col]:
                    self._add_error_to_one_delta_edge(row, col)

    def _train_one_sample(self, Xi, yi):
        self.act_vals[0] = Xi
        self._prop_fwd()
        self.deltas[-1] = self.perf_fcn_p(
            yi, self.act_vals[-1]) / self.samples
        self._prop_back()
        self._add_error_to_delta_edges()

    def _update_LR(self, batch_perf):
        #simple opt algorithms can get stuck without thit >1 multiplication
        epsilon = 10**-10
        if batch_perf < self.best_perf * (1+epsilon):
            self.LR *= 1.05
            self.best_perf = batch_perf
            self.best_weights = helper.copy_weights(self.weights)
        else:
            self.LR *= .7
            self.weights = helper.copy_weights(self.best_weights)
        if self.verb > 0:
            print(batch_perf, self.LR)

    def _test_batch(self, X, Y):
        batch_perf = 0
        for i in range(len(X)):
            self.act_vals[0] = X[i]
            self._prop_fwd()
            batch_perf += self.perf_fcn(Y[i], self.act_vals[-1]) / self.samples
        self._update_LR(batch_perf)
        return batch_perf

    def _verify_x(self, X):
        if X.ndim < 2:
            X = np.atleast_2d(X).T
        if X.shape[1] != self.layers[0]:
            print("input size %d, needed %d" % (X.shape[1], self.layers[0]))
            return False
        return X

    def _assign_batches(self, batch_type):
        if batch_type == helper.SOLO:
            self.num_batches = self.samples
            self.batch_size = 1
        if batch_type == helper.GROUP:
            self.num_batches = 1
            self.batch_size = self.samples
        if batch_type > 0:
            self.num_batches = int(self.samples ** (1 / batch_type))
            self.batch_size = int(np.ceil(self.samples / self.num_batches))

    def _re_init_once(self, X, Y):
        self._run_epoch(X, Y)
        if self.best_perf < self.gb_perf:
            self.gb_perf = self.best_perf
            self.gb_weights = helper.copy_weights(self.best_weights)
            self.gb_LR = self.LR

    def _init_k_times(self, X, Y, re_init, re_init_d, LR):
        for init_idx in range(re_init):
            self._initweights(std=2 * 10 ** (init_idx / re_init))
            self.best_perf = np.inf
            self.LR = LR
            for _ in range(re_init_d):
                self._re_init_once(X, Y)

    def _calc_norm_Y(self, Y):
        scale = np.amax(Y) - np.amin(Y)
        Y = Y / scale
        offset = np.amin(Y)
        Y = Y - offset
        # At this point, Y is [0,1]
        if self.trans[-1] in [helper.sigmoid]:
            pass
        if self.trans[-1] in [helper.tanh]:
            scale = scale / 2
            offset += 0.5
        return scale, offset

    def _normalize_Y(self, Y):
        return (Y / self.Yscale) - self.Yoffset

    def _denormalize_Y(self, Y):
        return (Y + self.Yoffset) * self.Yscale

    def _prepare_training(self, X, Y, LR,
                          batch_type, verb, nudge, objective, del_thresh):
        X = self._verify_x(X)
        if isinstance(X, bool):
            print("X size %d, need %d" % (X.shape[0], self.layers[0]))
            return
        if self.Xstd == []:
            X, self.Xstd, self.Xoffset = helper.normalize(X)
        else:
            X = helper.apply_norm(X, self.Xstd, self.Xoffset)
        self.samples = X.shape[0]
        self.verb = verb
        self.objective = objective
        self.del_thresh = del_thresh
        self.max_nudge = nudge
        self.nudge = 0
        self.best_epoch = 0
        self.LR = LR
        if self.Yscale == 0:
            self.Yscale, self.Yoffset = self._calc_norm_Y(Y)
        Y = self._normalize_Y(Y)
        self._assign_batches(batch_type)
        self.best_weights = helper.copy_weights(self.weights)
        self.gb_weights = helper.copy_weights(self.best_weights)
        self.gb_perf = np.copy(self.best_perf)
        self.gb_LR = self.LR
        return X, Y

#This is a very complicated function. Essentially if the
#optimization is stuck at a flat area, this should nudge it around randomly
#the math is to control the size of the adjustment to be random but relative
#to the edge weights
    def _nudge(self):
        self.nudge += 1
        for row in range(len(self.weights)):
            for col in range(len(self.weights[0])):
                if self.Lmap[row][col]:
                    self.weights[row][col] -= (
                            self.weights[row][col] *
                            np.random.random(self.weights[row][col].shape)
                            * (np.random.random())**2 * 0.1
                    )

    def _eval_perf(self, perf, epoch, del_thresh, max_fail):
        if perf < self.objective:
            print("reached optimization objective at \
                    {0} epochs".format(epoch))
            return helper.STOP_TRAIN
        if perf < self.gb_perf - del_thresh:
            self.nudge = 0
            self.gb_perf = perf
            self.best_epoch = epoch
        else:
            if epoch > self.best_epoch + max_fail:
                if self.nudge < self.max_nudge:
                    self._nudge()
                else:
                    print ("no longer improvng after {0} epochs"
                           .format(self.best_epoch))
                    return helper.STOP_TRAIN

    def train(self, X, Y, LR=10**-3, batch_type=helper.GROUP, verb=0,
              re_init=3, re_init_d=10, epochs = 10,
              nudge = 0, objective = 0, del_thresh=0, max_fail = np.inf):
        X, Y = self._prepare_training(X, Y, LR, batch_type, verb,
                                      nudge, objective, del_thresh)
        self._init_k_times(X, Y, re_init, re_init_d, LR)
        self.best_perf = self.gb_perf
        self.best_weights = helper.copy_weights(self.gb_weights)
        self.weights = helper.copy_weights(self.best_weights)
        self.LR = self.gb_LR
        for epoch in range(epochs):
            perf = self._run_epoch(X, Y)
            if self._eval_perf(perf, epoch, del_thresh, max_fail) == \
                    helper.STOP_TRAIN:
                break
        self.weights = helper.copy_weights(self.best_weights)

    def predict(self, x):
        x = helper.apply_norm(x, self.Xstd, self.Xoffset)
        self.samples = x.shape[0]
        output = np.empty([self.samples, self.act_vals[-1].shape[0]])
        # this can be made more efficient through matrix math.
        for d_idx in range(self.samples):
            self.act_vals[0] = np.atleast_1d(x[d_idx])
            self._prop_fwd()
            output[d_idx] = self.act_vals[-1]
        output = self._denormalize_Y(output)
        return output
