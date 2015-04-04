__author__ = 'mike'

from time import time
import cPickle as pickle
import sys

import numpy as np
import theano
from theano import tensor as T
import matplotlib.pyplot as plt
from theano.tensor.nnet import softmax

from utils import adam, quadratic_loss, variance, float32, int32, relu, sgd


rng = np.random.RandomState(int(time()))

sys.setrecursionlimit(100000)

floatX = theano.config.floatX


class ClockworkLayer(object):
    def __init__(self, groups_num, group_size, input_size, group_labels=None, activation_function=relu):
        self.groups_num = groups_num
        self.group_size = group_size
        self.input_size = input_size
        self.group_labels = group_labels if group_labels else 2 ** np.arange(groups_num)
        self.activation_function = activation_function

        self.W_in = theano.shared(np.random.normal(loc=0.0, scale=variance(input_size),
                                                   size=(input_size, groups_num, group_size)).astype(floatX))
        # name="{}_W_in".format(groups_num))

        # Weights for recurrent connection within the group

        self.W_self = np.random.normal(loc=0.0, scale=0.01,
                                       size=(groups_num * group_size, groups_num, group_size)).astype(floatX)
        self.W_self_nullifier = np.zeros(self.W_self.shape, dtype=floatX)
        for dx in xrange(groups_num * group_size):
            for g in xrange(groups_num):
                if g >= (dx // group_size):
                    self.W_self[dx][g] = 0.
                else:
                    self.W_self_nullifier[dx, g] = 1.
                    spng = rng.permutation(group_size)
                    self.W_self[dx][g][spng[15:]] = 0.

        self.W_self = theano.shared(self.W_self,
                                    name="{}_W_self".format(groups_num))
        # self.W_self = theano.shared(np.random.normal(loc=0.0, scale=0.01,
        # size=(groups_num * group_size, groups_num, group_size)).astype(
        # floatX),
        #     name="{}_W_self".format(groups_num))
        #


        self.biases = theano.shared(
            np.zeros((groups_num, group_size), dtype=floatX))

        self.initial_activation = theano.shared(np.random.normal(loc=0.0, scale=variance(groups_num * group_size),
                                                                 size=groups_num * group_size).astype(floatX),
                                                name='init_activation')

        self.params = [self.W_self, self.W_in, self.biases, self.initial_activation]
        self.timestep = theano.shared(1)

    def fprop(self, input_steps):
        if not getattr(self, '_fprop_func', None):
            theano_input = T.tensor3()
            theano_input.tag.test_value = np.ones((2, 10, 1), dtype=floatX)
            res, upd = self._fprop(theano_input)
            self._fprop_func = theano.function([theano_input], res, updates=upd)
        return self._fprop_func(input_steps)

    def _fprop(self, theano_input):
        def step(input_step, previous_activation, time_step, W_in, W_self, biases):
            new_activation = previous_activation.copy()
            modzero = T.nonzero(T.eq(T.mod(time_step, self.group_labels), 0))[0]
            W_in_now = T.flatten(W_in[:, modzero, :], outdim=2)
            W_self_now = T.flatten(W_self[:, modzero, :], outdim=2)
            biases_now = T.flatten(biases[modzero, :])
            activation = T.dot(input_step, W_in_now)
            activation += T.dot(previous_activation, W_self_now)
            activation += biases_now
            activation = self.activation_function(activation)
            modzero_activation_changes = (modzero * self.group_size) + (
                T.ones((modzero.shape[0], self.group_size), dtype='int32') * T.arange(self.group_size, dtype='int32')).T
            modzero_flatten = T.flatten(modzero_activation_changes).astype('int32')
            new_activation = T.set_subtensor(new_activation[:, modzero_flatten], activation)
            time_step += 1
            return new_activation, time_step

        initial_activation = T.ones((theano_input.shape[1], self.initial_activation.shape[0]),
                                    dtype=floatX) * self.initial_activation
        output_steps, updates = theano.scan(step, sequences=[theano_input],
                                            outputs_info=[initial_activation, self.timestep],
                                            non_sequences=[self.W_in, self.W_self, self.biases])
        output_steps = output_steps[0]
        scan_node = output_steps.owner.inputs[0].owner
        assert isinstance(scan_node.op, theano.scan_module.scan_op.Scan)
        n_pos = scan_node.op.n_seqs + 1
        init_h = scan_node.inputs[n_pos]
        self.init_h = init_h
        self.activation = output_steps
        return output_steps, updates


class OutputLayer(object):
    def __init__(self, shape, input_shape, activation_function=softmax):
        self.shape = shape
        self.input_shape = input_shape
        self.W_in = theano.shared(np.random.normal(loc=0.0, scale=variance(input_shape),
                                                   size=(input_shape, shape)).astype(floatX),
                                  name="output_W_in")
        self.biases = theano.shared(
            np.zeros(shape, dtype=floatX))  # np.random.normal(loc=0.0, scale=variance(input_shape),
        # size=shape).astype(floatX),
        # name="output_biases")

        self.params = [self.W_in, self.biases]
        self.activation_function = activation_function

    def _fprop(self, theano_input):
        def step(input_step, W, biases):
            activation = T.dot(input_step, W)
            activation += biases
            activation = self.activation_function(activation)
            return activation

        output_steps, updates = theano.scan(step, sequences=[theano_input],
                                            non_sequences=[self.W_in, self.biases])
        return output_steps, updates


class ClockworkRNN(object):
    def __init__(self, layer_specs=(1, (2, 20), (2, 10), 1), cost=quadratic_loss, update_fn=adam, learning_rate=0.001,
                 alpha=1.):
        self.alpha = theano.shared(float32(alpha))
        self.layer_specs = layer_specs
        self.layers = []
        previous_size = layer_specs[0]
        self.params = []
        self.cost = cost
        self.update_fn = update_fn
        self.learning_rate = theano.shared(float32(learning_rate))
        self.training_step = theano.shared(float32(1))

        for i, spec in enumerate(layer_specs[1:-1]):
            spec = [spec[0], spec[1], previous_size]
            self.layers.append(ClockworkLayer(*spec, activation_function=T.tanh))
            self.params.extend(self.layers[-1].params)
            previous_size = spec[0] * spec[1]
        self.layers.append(OutputLayer(layer_specs[-1], previous_size, T.tanh))
        self.params.extend(self.layers[-1].params)


    def fprop(self, input_steps):
        if not getattr(self, '_fprop_func', None):
            theano_input = T.tensor3()
            output, updates = self._fprop(theano_input)
            self._fprop_func = theano.function([theano_input], output, updates=updates)
        return self._fprop_func(input_steps)

    def _fprop(self, theano_input):
        theano_input = theano_input.dimshuffle(1, 0, 2)
        previous_input = theano_input
        updates = {}
        preq_reg_params = []
        # theano_input.tag.test_value = np.ones((2, 10, 1), dtype=floatX)
        for layer in self.layers[:-1]:
            previous_input, upd = layer._fprop(previous_input)
            updates.update(dict(upd))
            preq_reg_params.append([previous_input, layer.init_h])
        previous_input, upd = self.layers[-1]._fprop(previous_input)
        updates.update(dict(upd))
        output = previous_input.dimshuffle(1, 0, 2)
        self.preq_reg_params = preq_reg_params
        return output, updates

    def bptt(self, X, y):
        if not getattr(self, '_bptt', None):
            test_vals = create_batch_func_params(10, 0.1, 2)
            theano_input = T.tensor3('x')
            labeled_data = T.tensor3('y')
            theano_input.tag.test_value = test_vals[0]
            labeled_data.tag.test_value = test_vals[1]
            prediction, updates = self._fprop(theano_input)
            loss = self.cost(prediction, labeled_data)
            grads = T.grad(loss, self.params + sum(self.preq_reg_params, []))
            reg_grads = grads[len(self.params):]
            grads = grads[:len(self.params)]
            # for i, (layer, preq) in enumerate(zip(self.layers[:-1], self.preq_reg_params)):
            #     reg_grad = reg_grads[2 * i:2 * i + 2]
            #     if layer.activation_function == T.tanh:
            #         tmp_x = reg_grad[1][1:] * (1 - preq[0] ** 2)
            #     else:
            #         tmp_x = reg_grad[1][1:]
            #
            #     sh0 = tmp_x.shape[0]
            #     sh1 = tmp_x.shape[1]
            #     sh2 = tmp_x.shape[2]
            #     tmp_x = tmp_x.reshape((sh0 * sh1, sh2))
            #     tmp_x = T.dot(tmp_x, (layer.W_self * layer.W_self_nullifier).reshape((sh2, -1)))
            #     tmp_x = (tmp_x.reshape((sh0, sh1, sh2)) ** 2).sum(2)
            #     tmp_y = (reg_grad[1][1:] ** 2).sum(2)
            #     tmp_reg = (T.switch(T.ge(tmp_y, 1e-20), tmp_x / tmp_y, 1) - 1.) ** 2
            #     n_elems = T.mean(T.ge(tmp_y, 1e-20), axis=1)
            #     tmp_reg = tmp_reg.mean(1).sum() / n_elems.sum()
            #     tmp_gWhh = T.grad(tmp_reg, layer.W_self)
            #     grads[4 * i] += self.alpha * T.exp(self.training_step/-1000.0) * tmp_gWhh

            updates, norm, grad_threshold, momentum_norm, momentum_norm_threshold = self.update_fn(loss, self.params,
                                                                                                   grads=grads,
                                                                                                   learning_rate=self.learning_rate, training_step=self.training_step)
            updates.append((self.training_step, self.training_step + 1))
            self._bptt = theano.function([theano_input, labeled_data],
                                         [loss, norm, grad_threshold, momentum_norm, momentum_norm_threshold],
                                         updates=updates, allow_input_downcast=True)
            # self._bptt.trust_input = True
        return self._bptt(X, y)


def func_to_learn(X):
    return np.cos(np.sin(np.cos(X) + 1) + 1)


def create_batch_func_params(input_length=300, freq_var=0.1, size=20):
    freqs = float32(np.abs(np.random.normal(scale=freq_var, size=size)) + 0.1)
    # freqs = np.ones(size, dtype=floatX) * float32(0.1)
    X = np.array([np.ones(input_length, dtype=floatX) * freq for freq in freqs], dtype=floatX)[:, :, np.newaxis]
    x_series = np.array([np.linspace(0, input_length * freq, num=input_length, dtype=floatX) for freq in freqs],
                        dtype=floatX)
    y = func_to_learn(x_series).astype(floatX)[:, :, np.newaxis]
    return X, y, x_series


def main():
    net = ClockworkRNN((1, (4, 30), 1), update_fn=adam, learning_rate=0.001)
    # ones = np.ones((2, 10, 1), dtype=floatX)
    # res = net.fprop(ones)
    losses = []
    lrs = []
    norms, momentum_norms = [], []
    norms_thres, momentum_norms_thres = [], []
    # X, y, x_series = create_batch_func_params(500, 0.1, 2)
    X, y, x_series = create_batch_func_params(410, 0.1, 2000)
    best = np.inf
    last_best_index = 0
    decrement = float32(0.99)
    for i in range(8000):
        start = time()
        loss, norm, norm_theshold, momentum_norm, momentum_norm_threshold = net.bptt(X, y)
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

    with open('rnn.pickle', 'wb') as f:
        pickle.dump(net, f)

    # plt.figure(figsize=(40, 20), dpi=100)
    fig, ax1 = plt.subplots(figsize=(30, 10))
    ax2 = ax1.twinx()

    ax1.plot(losses, label='loss')
    ax1.set_ylabel('Loss')

    ax2.plot(lrs, color="red", label='learning rate')
    ax2.set_ylabel('Learning rate')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')

    fig.savefig('rnn_learning_rate_and_cost.jpg')
    plt.clf()

    fig, ax1 = plt.subplots(figsize=(30, 10))
    ax2 = ax1.twinx()

    ax1.plot(losses[-1000:], label='loss')
    ax1.set_ylabel('Loss')

    ax2.plot(lrs[-1000:], color="red", label='learning rate')
    ax2.set_ylabel('Learning rate')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')

    fig.savefig('rnn_learning_rate_and_cost_last1000.jpg')
    plt.clf()

    # NORMS
    fig, ax1 = plt.subplots(figsize=(30, 10))
    ax2 = ax1.twinx()

    ax1.plot(norms, label="norm", color='red')
    ax1.plot(norms_thres, label="norm threshold", color='blue')
    ax1.set_ylabel('Gradient norm')

    ax2.plot(momentum_norms, label="momentum norms", color='green')
    ax2.plot(momentum_norms_thres, label="momentum norms threshold", color='purple')
    ax2.set_ylabel('Momentum norm')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')
    fig.savefig('rnn_norms.jpg')
    plt.clf()

    # X, y, x_series = X[:10], y[:10], x_series[:10]
    X, y, x_series = create_batch_func_params(1500, 0.1, 10)

    prediction = net.fprop(X)
    # plt.close('all')
    fig, axarr = plt.subplots(len(X), sharex=True, figsize=(160, 5 * X.shape[0]))
    for i in range(len(X)):
        # axarr[i].figure.figure = plt.figure(figsize=(150, 12), dpi=100)
        axarr[i].set_title('freq:' + str(X[i][i]))
        axarr[i].plot(prediction[i], label='model', color='blue')
        axarr[i].plot(y[i], label='actual', color='green')
        axarr_twin = axarr[i].twinx()
        axarr_twin.plot(np.abs(y[i] - prediction[i]), label='error', color='red')
        h1, l1 = axarr[i].get_legend_handles_labels()
        h2, l2 = axarr_twin.get_legend_handles_labels()
        axarr[i].legend(h1 + h2, l1 + l2, loc=0)

    fig.savefig('rnn_prediction_vs_actual.jpg')

    print "done"


if __name__ == "__main__":
    main()