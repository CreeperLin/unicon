import torch
import numpy as np


# isaacgym/python/isaacgym/torch_utils.py
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def to_tensor(obj, **kwargs):
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, np.ndarray):
        x = torch.from_numpy(obj)
    else:
        x = torch.tensor(obj)
    return x.to(**kwargs)


def pp_tensor(x):
    return list(map(lambda e: round(e, 3), x.flatten().tolist()))


def torch_load_jit(model_path, env_cfg=None, train_cfg=None, device='cpu', traced=False):
    model = torch.jit.load(model_path).to(device=device)
    # model.requires_grad_(False)
    print(model)
    return model


def torch_load(model_path, env_cfg=None, train_cfg=None, traced=False, script=True, device='cpu'):
    model = torch.load(model_path, map_location=device)
    model.requires_grad_(False)
    print(model)
    if script:
        model = torch.jit.script(model)
    return model


def torch_no_grad():
    torch.set_grad_enabled(False)
    inference_mode = getattr(torch, 'inference_mode', None)
    if inference_mode is None:
        return
    global _torch_inference
    _torch_inference = inference_mode()
    _torch_inference.__enter__()


def torch_no_profiling():
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)


def torch_trace_func(func, inputs=tuple(), dct=None, module=True):
    dct = {} if dct is None else dct
    if module:

        class M(torch.nn.Module):
            pass

        mod = M()
        for k, v in dct.items():
            setattr(mod, k, v)
        _fn = func
        mod.forward = _fn
        func = mod
    func = torch.jit.trace(
        func,
        inputs,
        check_trace=False,
    )
    print(func.code)

    return func
