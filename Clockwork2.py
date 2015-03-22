__author__ = 'mike'

import numpy as np
import theano
from theano import tensor as T
import matplotlib.pyplot as plt
from theano.tensor.nnet import softmax

from utils import adam, quadratic_loss, variance


floatX = theano.config.floatX


def float32(k):
    return np.cast['float32'](k)


def int32(k):
    return np.cast['int32'](k)


def relu(x):
    return T.maximum(0, x)


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
        self.W_self = theano.shared(np.random.normal(loc=0.0, scale=variance(groups_num * group_size),
                                                     size=(groups_num * group_size, groups_num, group_size)).astype(
            floatX))

        # name="{}_W_self".format(groups_num))
        self.biases = theano.shared(
            np.zeros((groups_num, group_size), dtype=floatX))

        self.initial_activation = theano.shared(np.random.normal(loc=0.0, scale=variance(groups_num * group_size),
                                                                 size=groups_num * group_size).astype(floatX))

        self.params = [self.W_in, self.W_self, self.biases, self.initial_activation]
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

        initial_activation = T.ones((theano_input.shape[0], self.initial_activation.shape[0])) * self.initial_activation
        theano_input = theano_input.dimshuffle(1, 0, 2)
        output_steps, updates = theano.scan(step, sequences=[theano_input],
                                            outputs_info=[initial_activation, self.timestep],
                                            non_sequences=[self.W_in, self.W_self, self.biases])
        output_steps = output_steps[0].dimshuffle(1, 0, 2)
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

        theano_input = theano_input.dimshuffle(1, 0, 2)
        output_steps, updates = theano.scan(step, sequences=[theano_input],
                                            non_sequences=[self.W_in, self.biases])
        output_steps = output_steps.dimshuffle(1, 0, 2)
        return output_steps, updates


class ClockworkRNN(object):
    def __init__(self, layer_specs=(1, (2, 20), (2, 10), 1), cost=quadratic_loss, update_fn=adam, learning_rate=0.002):
        self.layer_specs = layer_specs
        self.layers = []
        previous_size = layer_specs[0]
        self.params = []
        self.cost = cost
        self.update_fn = update_fn
        self.learning_rate = theano.shared(float32(learning_rate))

        for i, spec in enumerate(layer_specs[1:-1]):
            spec = [spec[0], spec[1], previous_size]
            self.layers.append(ClockworkLayer(*spec))
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
        previous_input = theano_input
        updates = {}
        for layer in self.layers:
            theano_input.tag.test_value = np.ones((2, 10, 1), dtype=floatX)
            previous_input, upd = layer._fprop(previous_input)
            updates.update(dict(upd))
        return previous_input, updates

    def bptt(self, X, y):
        if not getattr(self, '_bptt', None):
            theano_input = T.tensor3()
            labeled_data = T.tensor3()
            prediction, updates = self._fprop(theano_input)
            loss = self.cost(prediction, labeled_data)
            updates = self.update_fn(loss, self.params, learning_rate=self.learning_rate)
            self._bptt = theano.function([theano_input, labeled_data], [loss], updates=updates)
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
    net = ClockworkRNN((1, (10, 30), (2, 10), 1))
    # ones = np.ones((2, 10, 1), dtype=floatX)
    # res = net.fprop(ones)
    losses = []
    X, y, x_series = create_batch_func_params()
    best = np.inf
    last_best_index = 0
    decrement = float32(0.95)

    for i in range(2000):
        losses.append(net.bptt(X, y))
        print i, ':', losses[-1]

        if best > losses[-1]:
            last_best_index = i
            best = losses[-1]
        elif i - last_best_index > 20:
            best = losses[-1]
            new_rate = net.learning_rate.get_value() * decrement
            net.learning_rate.set_value(new_rate)
            last_best_index = i
            print("New learning rate", new_rate)

    plt.plot(losses)
    plt.savefig('rnn_quadratic_cost.jpg')
    plt.clf()
    prediction = net.fprop(X)
    plt.figure(figsize=(150, 12), dpi=100)
    plt.plot(prediction[0], label='model')
    plt.plot(y[0], label='actual')
    plt.legend()
    plt.savefig('rnn_prediction_vs_actual.jpg')
    print "done"


if __name__ == "__main__":
    main()