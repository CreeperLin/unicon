import numpy as np


def data_preproc(d,):
    if isinstance(d, dict):
        return {k: data_preproc(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [data_preproc(v) for v in d]
    elif isinstance(d, np.ndarray):
        return d
    elif isinstance(d, (int, float, str, bool)):
        return d
    else:
        return str(d)


def load(path=None):
    rec = np.load(path, allow_pickle=True).item()
    return rec


def save(path=None, data=None):
    np.save(path, data_preproc(data), allow_pickle=True)
