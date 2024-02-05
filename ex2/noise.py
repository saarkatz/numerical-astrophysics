import numpy as np


def apply_uniform_noise(y, amplitude):
    return y + np.random.uniform(-amplitude, amplitude, y.shape)


def apply_gaussian_noise(y, amplitude, mean, std_dev):
    return y + amplitude * np.random.normal(mean, std_dev, y.shape)

