__author__ = 'mike'

import cPickle as pickle
from time import time
import numpy as np
import theano
from Clockwork2 import ClockworkRNN
from utils import adam, quadratic_loss, variance, float32, int32, relu, sgd, cross_entropy

floatX = theano.config.floatX


def hot_to_sparse(data, size=88):
    result = []
    for a in data:
        song = np.zeros((1788, size), dtype=floatX)
        for i, b in enumerate(a):
            # offset
            b = np.array(b, dtype='int32') - 21
            song[i, b] = 1
        result.append(np.asarray(song))
    return np.array(result, dtype=floatX)


def main():
    nottingham = pickle.load(file("Nottingham.pickle"))
    train = hot_to_sparse(nottingham['train'][:20], 88)
    net = ClockworkRNN((88, (4, 30), 88), update_fn=adam, learning_rate=0.001, cost=quadratic_loss)
    losses = []
    lrs = []
    norms, momentum_norms = [], []
    norms_thres, momentum_norms_thres = [], []
    # X, y, x_series = create_batch_func_params(500, 0.1, 2)
    best = np.inf
    last_best_index = 0
    decrement = float32(0.99)
    for i in range(2000):
        start = time()
        loss, norm, norm_theshold, momentum_norm, momentum_norm_threshold = net.bptt(train[:, :-1], train[:, 1:])
        losses.append(loss)
        norms.append(norm)
        momentum_norms.append(momentum_norm)
        norms_thres.append(float32(norm_theshold))
        momentum_norms_thres.append(float32(momentum_norm_threshold))
        lrs.append(net.learning_rate.get_value())
        epoch_time = time() - start
        print i, ':', losses[-1], " took :", epoch_time

        if best > losses[-1]:
            last_best_index = i
            best = losses[-1]
        elif i - last_best_index > 20:
            best = losses[-1]
            new_rate = net.learning_rate.get_value() * decrement
            net.learning_rate.set_value(new_rate)
            last_best_index = i
            print("New learning rate", new_rate)


if __name__ == "__main__":
    main()