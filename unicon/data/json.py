import json
import numpy as np


def load(path=None,):
    with open(path, 'r') as f:
        rec = json.load(f)
    rec = {k: (np.array(v) if isinstance(v, list) else v) for k, v in rec.items()}
    return rec


def save(
    path=None,
    data=None,
):
    with open(path, 'w') as f:
        json.dump(f, data)
