_factories = {}


def register(name, cls):
    _factories[name] = cls


def get_factory(fmt):
    cls = _factories[fmt]
    return cls


def serializer_json(round=3, to_dtype='float32'):
    import json
    import numpy as np

    dtype = np.dtype(to_dtype)

    def dump_fn(states):
        data = {k: s.astype(np.float64).round(round).tolist() for k, s in states.items()}
        return json.dumps(data, separators=(",", ":")).encode()

    def load_fn(blob):
        raw = json.loads(blob.decode())
        return {k: np.array(v, dtype=dtype) for k, v in raw.items()}

    return dump_fn, load_fn


def serializer_msgpack(map_dtypes=True):
    import numpy as np
    import msgpack
    import msgpack_numpy as m

    m.patch()
    packer = msgpack.Packer(use_bin_type=True)
    unpacker = msgpack.Unpacker(raw=False)

    default_dtype_mappings = {
        # np.float64: np.float16,
        np.float64: np.float32,
        np.float32: np.float16,
    }

    dtype_mappings = default_dtype_mappings.copy()
    if map_dtypes is None:
        dtype_mappings = {}
        map_dtypes = {}
    elif map_dtypes is True:
        map_dtypes = {}
    dtype_mappings.update(map_dtypes)
    dtype_mappings = {np.dtype(k): np.dtype(v) for k, v in dtype_mappings.items()}
    dtype_mappings_rev = {v: k for k, v in dtype_mappings.items()}
    print('dtype_mappings', dtype_mappings)
    print('dtype_mappings_rev', dtype_mappings_rev)

    def encode(obj):
        if not len(dtype_mappings):
            return obj
        if isinstance(obj, np.ndarray):
            dtype = dtype_mappings.get(obj.dtype)
            if dtype is not None:
                obj = obj.astype(dtype, copy=False)
            return obj
        elif isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        else:
            return obj

    def decode(obj):
        if not len(dtype_mappings):
            return obj
        if isinstance(obj, np.ndarray):
            dtype = dtype_mappings_rev.get(obj.dtype)
            if dtype is not None:
                obj = obj.astype(dtype, copy=False)
            return obj
        elif isinstance(obj, dict):
            return {k: decode(v) for k, v in obj.items()}
        else:
            return obj

    def dump_fn(x):
        return packer.pack(encode(x))

    def load_fn(y):
        unpacker.feed(y)
        try:
            ret = unpacker.unpack()
            # return ret
            return decode(ret)
        except Exception as e:
            print('serializer_msgpack exc', e)
        return None
        # unpacker._consume()

    return dump_fn, load_fn


def serializer_json_b64(to_dtype='float32'):
    import io
    import json
    import base64
    import numpy as np

    buf = io.BytesIO()
    dtype = np.dtype(to_dtype)

    def dump_fn(states):
        out = {}
        for k, arr in states.items():
            buf.seek(0)
            buf.truncate(0)
            np.save(buf, arr.astype(dtype))
            out[k] = base64.b64encode(buf.getvalue()).decode()
        return json.dumps(out).encode()

    def load_fn(blob):
        raw = json.loads(blob.decode())
        out = {}
        for k, b64 in raw.items():
            data = base64.b64decode(b64)
            out[k] = np.load(io.BytesIO(data)).astype(dtype)
        return out

    return dump_fn, load_fn


def serializer_npz():
    import io
    import numpy as np

    buf = io.BytesIO()

    def dump_fn(states):
        buf.seek(0)
        buf.truncate(0)
        np.savez_compressed(buf, **states)
        return buf.getvalue()

    def load_fn(blob):
        return dict(np.load(io.BytesIO(blob)))

    return dump_fn, load_fn


_factories.update({k[k.index('_') + 1:]: v for k, v in globals().items() if k.startswith('serializer_')})


def main():
    import numpy as np
    import timeit
    states = {
        "pos": np.random.randn(32).astype(dtype=np.float32),
        "vel": np.random.randn(32).astype(dtype=np.float32),
        "acc": np.random.randn(32).astype(dtype=np.float32),
    }

    number = 5000
    for fmt in ["msgpack", "json", "json_b64", "npz"]:
        print(f"\n--- {fmt} ---")

        dump, load = get_factory(fmt)()
        blob = dump(states)
        print('dump', timeit.timeit(lambda: dump(states), number=number) / number)
        print(f"dump size: {len(blob)} bytes")

        restored = load(blob)
        print('load', timeit.timeit(lambda: load(blob), number=number) / number)

        for k in states:
            a = states[k]
            b = restored[k]
            if a.shape != b.shape or a.dtype != b.dtype:
                print(f"ERROR: shape/dtype mismatch for '{k}'")
                print("original:", a.shape, a.dtype)
                print("restored:", b.shape, b.dtype)

            diff = a - b
            max_err = np.abs(diff).max()
            print(k, a.shape, np.linalg.norm(diff), max_err)


if __name__ == "__main__":
    main()
