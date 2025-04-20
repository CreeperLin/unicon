import numpy as np


def sampler_uniform(low, high, rng=None, num_samples=100, repeats=100, seed=1, **kwds):
    rng = np.random.RandomState(seed) if rng is None else rng
    low = np.array(low)
    high = np.array(high)
    t = 0
    while True:
        if num_samples is not None and t >= num_samples:
            return
        v = rng.uniform(low, high)
        for i in range(repeats):
            yield v
            t += 1


def sampler_sine(low, high, num_samples=100, freq=1, dt=0.02):
    low = np.array(low)
    high = np.array(high)
    span = high - low
    freq = freq * dt * 2 * np.pi
    t = 0
    while True:
        if num_samples is not None and t >= num_samples:
            return
        a = np.sin(t * freq)
        v = low + span * (a + 1) * 0.5
        yield v
        t += 1


def sampler_waveform(low, high, num_samples=100, freq=1, dt=0.02, wave_type='square', wave_kwds=None):
    from scipy.signal import waveforms
    wave_fn = getattr(waveforms, wave_type)
    wave_kwds = {} if wave_kwds is None else wave_kwds
    low = np.array(low)
    high = np.array(high)
    span = high - low
    freq = freq * dt * 2 * np.pi
    t = 0
    while True:
        if num_samples is not None and t >= num_samples:
            return
        a = wave_fn(t * freq, **wave_kwds)
        v = low + span * (a + 1) * 0.5
        yield v
        t += 1


from functools import partial

sampler_triangle = partial(sampler_waveform, wave_type='sawtooth', wave_kwds=dict(width=0.5))
