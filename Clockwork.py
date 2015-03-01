__author__ = 'mike'

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
floatX = theano.config.floatX

def variance(input_size):
    return 2.0/input_size

class ClockworkGroup(object):

    def __init__(self, size, label_number, input_shape, greater_label_shapes,
                 shorter_label_shapes=None,
                 initial_activation=None,
                 activation_function=sigmoid):
        self.size = size
        self.input_shape = input_shape
        self.label = label_number
        self.sl_groups_sizes = shorter_label_shapes
        self.gl_groups_sizes = greater_label_shapes
        self.params = []
        if initial_activation:
            self.current_activation = theano.shared(initial_activation)
        else:
            self.current_activation = theano.shared(np.zeros(size, dtype=floatX))
        self.act_func = activation_function

    def create_params(self):
        if self.params:
            return self.params

        # Input weight from previous layer/inpuy
        self.W_in = theano.shared(np.random.normal(loc=0.0, scale=variance(self.input_shape),
                                                   size=(self.size, self.input_shape)))

        # Weights for recurrent connection within the group
        self.W_self = theano.shared(np.random.normal(loc=0.0, scale=variance(self.size),
                                                     size=(self.size, self.size)))

        # Weights from layers with bigger label than the current one
        self.W_h_inc = []
        for gl_size in self.gl_groups:
            self.W_h_inc.append(theano.shared(np.random.normal(loc=0.0, scale=variance(gl_size),
                                                               size=(self.size, gl_size))))
        self.biases = theano.shared(np.random.normal(loc=0.0, scale=variance(self.size),
                                                     size=self.size))

        self.params.append(self.W_in)
        self.params.append(self.W_self)
        self.params.extend(self.W_h_inc)
        self.params.append(self.biases)
        return self.params

    def get_activation(self, previous_layer_activation, greater_group_activation, time_step):
        activation = T.dot(self.W_in, previous_layer_activation)
        activation += sum([T.dot(p_w, p_a) for p_w, p_a in zip(self.W_h_inc, greater_group_activation)])
        activation += T.dot(self.W_self, self.current_activation)
        activation += self.biases
        activation = self.act_func(activation)
        return T.switch(T.eq(time_step % self.label, 0), activation, 0), activation

class ClockworkLayer(object):

    def __init__(self, number_of_groups, neuron_array, label_array, input_shape):
        self.n_groups = number_of_groups
        self.neuron_array = neuron_array
        self.label_array = label_array
        self.input_shape = input_shape
        self.groups = []

    def create_groups(self):
        for n, l in zip(self.neuron_array, self.label_array):
            self.groups.append(ClockworkGroup(n, l, self.input_shape))
