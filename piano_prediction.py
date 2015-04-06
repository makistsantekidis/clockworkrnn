__author__ = 'mike'

import cPickle as pickle
from time import time
import numpy as np
import theano
from Clockwork import ClockworkRNN
from utils import adam, quadratic_loss, variance, float32, int32, relu, sgd, cross_entropy

floatX = theano.config.floatX


def hot_to_sparse(data, size=88):
    result = []
    length = 254
    for a in data:
        song = np.zeros((length, size), dtype=floatX)
        for i, b in enumerate(a[:length]):
            # offset
            b = np.array(b, dtype='int32') - 21
            song[i, b] = 1
        result.append(np.asarray(song))
    return np.array(result, dtype=floatX)


def main():
    nottingham = pickle.load(file("Nottingham.pickle"))
    train = hot_to_sparse(nottingham['train'], 88)
    print len(train)
    np.random.shuffle(train)
    net = ClockworkRNN((88, (4, 30), 88), update_fn=adam, learning_rate=0.001, cost=quadratic_loss)
    batch_size = 700
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
        closs, cnorm, cnorm_threshold, cmomentum_norm, cmomentum_norm_threshold = [], [], [], [], []
        for g in xrange(0, len(train), batch_size):
            loss, norm, norm_theshold, momentum_norm, momentum_norm_threshold = net.bptt(train[g:g+batch_size, :-1], train[g:g+batch_size, 1:])
            closs.append(loss)
            cnorm.append(norm)
            cnorm_threshold.append(norms_thres)
            cmomentum_norm.append(momentum_norm)
            cmomentum_norm_threshold.append(momentum_norm_threshold)

        losses.append(np.mean(closs))
        norms.append(np.mean(cnorm))
        momentum_norms.append(np.mean(cnorm_threshold))
        norms_thres.append(np.mean(float32(cnorm_threshold)))
        momentum_norms_thres.append(np.mean([float32(cnmt) for cnmt in cmomentum_norm_threshold]))
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