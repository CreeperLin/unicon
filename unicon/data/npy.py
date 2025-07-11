import numpy as np


def load(path=None):
    rec = np.load(path, allow_pickle=True).item()
    return rec


def save(path=None, data=None):
    np.save(path, data, allow_pickle=True)
