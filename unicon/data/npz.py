import numpy as np


def load(path=None):
    rec = np.loadz(path)
    return dict(rec)


def save(path=None, data=None):
    np.savez(path, **data)
