__author__ = 'mike'

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import softmax
import matplotlib.pyplot as plt

floatX = theano.config.floatX


def variance(input_size):
    return 2.0 / input_size


def negative_log_likelihood(output, prediction):
    output, prediction = T.flatten(output), T.flatten(prediction)
    return -T.mean(T.log(output)[prediction])


class ClockworkGroup(object):
    def __init__(self, size, label_number, input_shape, greater_label_shapes,
                 shorter_label_shapes=None,
                 initial_activation=None,
                 activation_function=T.tanh):
        self.size = size
        self.input_shape = input_shape
        self.label = label_number
        self.sl_groups_sizes = shorter_label_shapes
        self.gl_groups_sizes = greater_label_shapes
        if initial_activation:
            self.current_activation = theano.shared(initial_activation)
        else:
            self.current_activation = theano.shared(np.zeros(size, dtype=floatX))
        self.act_func = activation_function
        self.params = self.create_params()

    def create_params(self):
        self.params = []
        # Input weight from previous layer/inpuy
        self.W_in = theano.shared(np.random.normal(loc=0.0, scale=variance(self.input_shape),
                                                   size=(self.size, self.input_shape)).astype(floatX),
                                  name="{}_W_in".format(self.label))

        # Weights for recurrent connection within the group
        self.W_self = theano.shared(np.random.normal(loc=0.0, scale=variance(self.size),
                                                     size=(self.size, self.size)).astype(floatX),
                                    name="{}_W_self".format(self.label))


        # Weights from layers with bigger label than the current one
        self.W_h_inc = []
        for i, gl_size in enumerate(self.gl_groups_sizes):
            self.W_h_inc.append(theano.shared(np.random.normal(loc=0.0, scale=variance(gl_size),
                                                               size=(self.size, gl_size)).astype(floatX),
                                              name="{}_{}_W_h_inc".format(i, self.label)))

        self.biases = theano.shared(np.zeros(self.size, dtype=floatX))#np.random.normal(loc=0.0, scale=variance(self.size),
                                                     # size=self.size).astype(floatX),
                                    # name="{}_biases".format(self.label))

        self.params.append(self.W_in)
        self.params.append(self.W_self)
        self.params.extend(self.W_h_inc)
        self.params.append(self.biases)
        return self.params

    def get_activation(self, previous_layer_activation, time_step, greater_group_activation=[],
                       current_activation=None):
        if not current_activation:
            current_activation = self.current_activation
        W_in, W_self, W_h_inc, biases = self.params[0], self.params[1], self.params[2:-1], self.params[-1]

        new_activation = T.dot(W_in, previous_layer_activation)
        new_activation += sum([T.dot(p_w, p_a) for p_w, p_a in zip(W_h_inc, greater_group_activation)])
        new_activation += T.dot(W_self, current_activation)
        new_activation += biases
        new_activation = self.act_func(new_activation)
        # return new_activation
        return T.switch(T.eq(time_step % self.label, 0), new_activation, current_activation)


class ClockworkLayer(object):
    def __init__(self, neuron_array, label_array, input_shape, activation_function=T.tanh):
        self.neuron_array = neuron_array
        self.label_array = label_array
        self.input_shape = input_shape
        self.groups = []
        self.params = []
        self.group_activations = []
        self.updates = []
        for i, (n, l) in enumerate(zip(neuron_array, label_array)):
            self.groups.append(ClockworkGroup(n, l, input_shape, neuron_array[i + 1:],
                                              activation_function=activation_function))
        self.create_optimization_parameters()


    def create_optimization_parameters(self):
        self.params = []
        for group in self.groups:
            self.params.extend(group.params)

    def get_current_group_activations(self):
        current_activations = []
        for group in self.groups:
            current_activations.append(group.current_activation)
        return current_activations

    def get_activations(self, input_activation, time_step, current_activation=None):
        self.group_activations = []
        if not current_activation:
            current_activation = []
            for group in self.groups:
                current_activation.append(group.current_activation)

        for i, group in reversed(list(enumerate(self.groups))):
            assert isinstance(group, ClockworkGroup)
            self.group_activations.append(
                group.get_activation(input_activation, time_step,
                                     self.group_activations,
                                     current_activation=current_activation[i]))

        self.group_activations = list(reversed(self.group_activations))
        # self.updates = list(zip(reversed(current_activations), self.group_activations))
        return self.group_activations
        # return T.concatenate(self.group_activations)


class InputLayer(object):
    def __init__(self, shape):
        self.shape = shape
        self.input_tensor = T.vector('X')

    def get_input(self, *args, **kwargs):
        return self.input_tensor


class OutputLayer(object):
    def __init__(self, shape, input_shape, activation_function=softmax):
        self.shape = shape
        self.input_shape = input_shape
        self.W_in = theano.shared(np.random.normal(loc=0.0, scale=variance(input_shape),
                                                   size=(shape, input_shape)).astype(floatX),
                                  name="output_W_in")
        self.bias = theano.shared(np.zeros(shape, dtype=floatX))#np.random.normal(loc=0.0, scale=variance(input_shape),
        #                                            size=shape).astype(floatX),
        #                           name="output_biases")

        self.params = [self.W_in, self.bias]
        self.activation_function = activation_function

    def get_activation(self, previous_layer_activation):
        return self.activation_function(T.dot(self.params[0], previous_layer_activation) + self.params[1])


class ClockWorkRNN(object):
    def __init__(self, layer_sizes=(1, ((10, 10, 10), (1, 2, 4)), 2),
                 cost=negative_log_likelihood,
                 hidden_activation=T.tanh,
                 output_activation=softmax):
        self.layer_sizes = layer_sizes
        self.cost = cost
        self.input_layer = InputLayer(layer_sizes[0])
        self.hidden_layers = []
        self.time_step = theano.shared(np.int32(1))
        previous_size = layer_sizes[0]
        previous_output = self.input_layer.get_input()
        self.params = []
        for neuron_array, label_array in layer_sizes[1:-1]:
            hidden_layer = ClockworkLayer(neuron_array, label_array, previous_size,
                                          activation_function=hidden_activation)
            previous_size = sum(neuron_array)
            self.params += hidden_layer.params
            self.hidden_layers.append(hidden_layer)
        self.output_layer = OutputLayer(layer_sizes[-1], previous_size, activation_function=output_activation)
        self.params += self.output_layer.params


        # theano tensors
        self.y = T.vector('y')
        self.Y = T.matrix('Y')
        self.X = self.input_layer.get_input()
        self.XT = T.matrix('XT')
        # test_values
        self.X.tag.test_value = np.random.randn(self.input_layer.shape)
        self.y.tag.test_value = np.asarray([1, 0], dtype=np.int32)
        self.Y.tag.test_value = np.asarray([[1, 0]] * 10, dtype=np.int32)
        self.XT.tag.test_value = np.arange(10, dtype=floatX).reshape((-1, 1))

    def _recurrence(self, x_slice, time_step, *hidden_activations):
        layer_activations = self._split_layer_activations(hidden_activations)
        incoming_input = x_slice
        new_hidden_activations = []
        for layer, activation in zip(self.hidden_layers, layer_activations):
            output = layer.get_activations(input_activation=incoming_input, time_step=time_step,
                                           current_activation=activation)
            new_hidden_activations.extend(output)
            incoming_input = T.concatenate(output)
        output = T.flatten(self.output_layer.get_activation(incoming_input))
        return [output], zip(hidden_activations, new_hidden_activations) + [(time_step, time_step + 1)]


    def _split_layer_activations(self, activations):
        layer_activations = []
        prev_layer_size = 0
        for layer in self.hidden_layers:
            layer_size = len(layer.neuron_array)
            layer_activations.append(activations[prev_layer_size:layer_size])
            prev_layer_size = layer_size
        return layer_activations

    def _get_current_hidden_activations(self):
        activations = []
        for layer in self.hidden_layers:
            activations.append(layer.get_current_group_activations())
        return sum(activations, [])


    def grad(self):
        return T.grad(self.cost(self.output, self.y), self.params)

    def fptt(self):
        result, updates = theano.scan(fn=self._recurrence, sequences=[self.XT],
                                      non_sequences=[self.time_step] + self._get_current_hidden_activations())
        return result, updates

    def reset(self):
        if not getattr(self, '_reset', None):
            activations = self._get_current_hidden_activations()
            time_step = self.time_step
            self._reset = theano.function([], [], updates=[(activation, T.zeros_like(activation)) for activation in
                                                           activations] +
                                                          [(time_step, np.int32(1))])
        return self._reset()

    def get_current_timestep(self):
        if not getattr(self, '_get_timestep', None):
            self._get_timestep = theano.function([], [self.time_step])
        return self._get_timestep()

    def get_current_activations(self):
        if not getattr(self, '_get_current_activations', None):
            self._get_current_activations = theano.function([],
                                                              outputs=self._get_current_hidden_activations())
        return self._get_current_activations()


def adam(loss, all_params, learning_rate=0.02, beta1=0.1, beta2=0.001,
         epsilon=1e-8, l_decay=1 - 1e-4):


    updates = []
    all_grads = theano.grad(loss, all_params)

    i = theano.shared(np.dtype(theano.config.floatX).type(0))
    i_t = i + 1.

    learning_rate_t = learning_rate * T.sqrt(1. - (1. - beta2) ** i_t) / (1. - (1. - beta1) ** i_t)

    beta1_t = 1 - (1 - beta1) * l_decay ** (i_t - 1)  # ADDED

    for param_i, g in zip(all_params, all_grads):
        m = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m)
        v_t = (beta2 * (g ** 2)) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param_i - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t))
    updates.append((i, i_t))
    return updates


def quadratic_loss(a, b):
    a, b = a.flatten(), b.flatten()
    # return T.mean(binary_crossentropy(a, b))
    return T.mean((b - a) ** 2)


if __name__ == "__main__":
    g1_size = 5
    net = ClockWorkRNN(layer_sizes=(1, ([100] * g1_size, [2**i for i in range(g1_size)]), 1),
                       cost=quadratic_loss,
                       hidden_activation=T.tanh,
                       output_activation=T.tanh)
    # grads = net.grads()

    res, upd = net.fptt()
    fptt = theano.function(inputs=[net.XT], outputs=res, updates=upd)
    # print f(net.XT.tag.test_value)
    # print net.Y.tag.test_value

    # grads = T.grad(net.cost(res[0], net.Y), net.params)

    predict_after = 10
    loss = net.cost(res[predict_after:-2], net.Y[predict_after+2:])

    train_updates = adam(loss, net.params, learning_rate=0.0002)
    train_func = theano.function(inputs=[net.XT, net.Y], outputs=[loss], updates=train_updates + upd)

    losses = []

    for i in range(500):
        X = np.linspace(0, np.abs(np.int32(np.random.randn()*13))+5, num=np.abs(np.int32(np.random.randn()*150))+80, dtype=floatX)
        y = np.sin(X).astype(floatX)
        losses.append(train_func(X.reshape(-1, 1), y.reshape(-1, 1))[0])
        # print net.get_current_activations()
        print losses[-1]
        net.reset()

    from pprint import pprint

    X = np.linspace(0, 30, num=3000, dtype=floatX)
    y = np.sin(X).astype(floatX)
    print ("Inaccuracy", np.abs(fptt(X.reshape(-1, 1)).reshape(-1) - y.reshape(-1)).mean())
    plt.plot(losses)
    plt.savefig('rnn_quadratic_cost.jpg')
    plt.clf()
    plt.plot(X, fptt(X.reshape(-1, 1)), label='model')
    plt.plot(X, np.sin(X), label='actual')
    plt.legend()
    plt.savefig('rnn_prediction_vs_actual.jpg')
