__author__ = 'mike'

import theano
from theano.ifelse import ifelse
from theano import tensor as T
import numpy as np


def float32(k):
    return np.cast['float32'](k)


def int32(k):
    return np.cast['int32'](k)


def relu(x):
    return T.maximum(0, x)


def variance(input_size):
    return 2.0 / input_size


def negative_log_likelihood(output, prediction):
    output, prediction = T.flatten(output), T.flatten(prediction)
    return -T.mean(T.log(output)[prediction])


def cross_entropy(output, prediction):
    return T.nnet.categorical_crossentropy(prediction, output).mean()


def sgd(loss, all_params, grads=None, const=[], learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
        gamma=1 - 1e-8, grad_threshold=10., momentum_threshold=1., ema_n=0.999, max_drawdown=0.4, drawdown_dumping=0.4,
        training_step=None):
    """
    Code taken from https://gist.github.com/skaae/ae7225263ca8806868cb

    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    grad_threshold = theano.shared(float32(grad_threshold))
    if grads:
        all_grads = grads
    else:
        all_grads = theano.grad(loss, all_params, consider_constant=const)

    grad_norm = T.sqrt(sum([(grad ** 2).sum() if grad.name != 'init_activationq' else 0 for grad in all_grads]))
    all_grads = ifelse(grad_norm > grad_threshold,
                       [(grad * grad_threshold) / grad_norm if grad.name != 'init_activationq' else grad for grad in
                        all_grads],
                       all_grads)

    updates.append((grad_threshold, ema_n * grad_threshold + (1 - ema_n) * grad_norm))

    for theta_previous, grad in zip(all_params, all_grads):
        updates.append((theta_previous, theta_previous - learning_rate * grad))

    return updates, grad_norm, grad_threshold, theano.shared(0), theano.shared(0)


def adam(loss, all_params, grads=None, const=[], learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1 - 1e-8, grad_threshold=20., momentum_threshold=.5, ema_n=0.991, max_drawdown=1.0,
         drawdown_dumping=.99, training_step=None):
    """
    Code taken from https://gist.github.com/skaae/ae7225263ca8806868cb

    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    param_updates = []
    grad_threshold = theano.shared(float32(grad_threshold))
    momentum_threshold = theano.shared(float32(momentum_threshold))
    if grads:
        all_grads = grads
    else:
        all_grads = theano.grad(loss, all_params, consider_constant=const)
    grad_norm = T.sqrt(sum([(grad ** 2).sum() if grad.name != 'init_activationq' else 0 for grad in all_grads]))
    all_grads = ifelse(grad_norm > grad_threshold,
                       [(grad * grad_threshold) / grad_norm if grad.name != 'init_activationq' else grad for grad in
                        all_grads],
                       all_grads)

    updates.append((grad_threshold, ema_n * grad_threshold + (1 - ema_n) * grad_norm))

    # for theta_previous, grad in zip(all_params, all_grads):
    # updates.append((theta_previous, theta_previous - learning_rate * grad))
    #
    # return updates, grad_norm


    l_previous = theano.shared(np.float32(0))
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1 * gamma ** (t - 1)  # (Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        m, v = m_previous, v_previous

        # m, v = ifelse((l_previous - loss) / l_previous < max_drawdown, [m, v],
        # [m * np.float32(drawdown_dumping), v * np.float32(drawdown_dumping)])

        # m, v = ifelse(T.eq(T.mod(training_step, 10), 0), [m, v],
        # [m * np.float32(drawdown_dumping), v * np.float32(drawdown_dumping)])

        m = b1_t * m + (1 - b1_t) * g  # (Update biased first moment estimate)
        v = b2 * v + (1 - b2) * g ** 2  # (Update biased second raw moment estimate)
        m_hat = m / (1 - b1 ** t)  # (Compute bias-corrected first moment estimate)
        v_hat = v / (1 - b2 ** t)  # (Compute bias-corrected second raw moment estimate)
        theta_update = (alpha * m_hat) / (T.sqrt(v_hat) + e)  # (Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        param_updates.append(theta_update)
    updates.append((t, t + 1.))
    updates.append((l_previous, loss))

    momentum_norm = T.sqrt(
        sum([(param ** 2).sum() if param.name != 'init_activation' else 0 for param in param_updates]))
    param_updates = ifelse(momentum_norm > momentum_threshold,
                           [(grad * momentum_threshold) / momentum_norm if grad.name != 'init_activation' else grad for
                            grad
                            in
                            param_updates],
                           param_updates)
    updates.append((momentum_threshold, ema_n * momentum_threshold + (1 - ema_n) * momentum_norm))

    for theta_previous, g in zip(all_params, param_updates):
        updates.append((theta_previous, theta_previous - g))

    return updates, grad_norm, grad_threshold, momentum_norm, momentum_threshold


def quadratic_loss(a, b):
    return T.mean((b - a) ** 2, axis=0).sum()


