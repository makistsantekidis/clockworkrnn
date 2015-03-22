__author__ = 'mike'

import numpy as np
import theano
from theano import tensor as T
from theano import printing

from utils import variance


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
            theano_input.tag.test_value = np.ones((1, 10, 1), dtype=floatX)
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
            printing.Print("modezero_flatten")(modzero_flatten)
            printing.Print("prev_activations")(new_activation)
            new_activation = T.set_subtensor(new_activation[:, modzero_flatten], activation)
            printing.Print("new_activations")(new_activation)
            time_step += 1
            return new_activation, time_step

        initial_activation = T.ones((theano_input.shape[0], self.initial_activation.shape[0])) * self.initial_activation
        theano_input = theano_input.dimshuffle(1, 0, 2)
        output_steps, updates = theano.scan(step, sequences=[theano_input],
                                            outputs_info=[initial_activation, self.timestep],
                                            non_sequences=[self.W_in, self.W_self, self.biases])
        return output_steps, updates


def main():
    net = ClockworkLayer(2, 5, 1)
    ones = np.ones((1, 10, 1), dtype=floatX)
    print net.initial_activation.eval()
    res = net.fprop(ones)
    print "done"


if __name__ == "__main__":
    main()