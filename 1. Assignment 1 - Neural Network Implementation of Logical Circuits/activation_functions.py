import numpy as np


def bi_sigmoid(x):
    # prevent exponential overflow
    clipped_neg_x = np.clip(-x, -500, 500)

    #  sigmoid(x) = 1 / (1 + exp(-x))
    return 1.0 / (1.0 + np.exp(clipped_neg_x))


def bi_sigmoid_derivative(y):
    return y * (1.0 - y)


def bip_sigmoid(x):
    #  biploar sigmoid(x) = 2 * sigma(x) - 1
    return 2.0 * bi_sigmoid(x) - 1.0


def bip_sigmoid_derivative(y):
    return 0.5 * (1.0 - y ** 2)


def sign_activation(y_in, y):
    """
    Discrete sign activation function: y(t+1) = sgn(y_in(t))
    - If y_in > 0: returns +1.0
    - If y_in < 0: returns -1.0
    - If y_in == 0: returns the previous state (y) to prevent oscillation.
    """
    if y_in > 0:
        return 1.0
    elif y_in < 0:
        return -1.0
    else:
        return float(y)  # Returns the previous state if y_in is exactly 0
