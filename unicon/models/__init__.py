import os


def load_model_torch(model_path, device=None):
    from unicon.utils.torch import torch_load_jit, torch_no_grad, torch_no_profiling
    torch_no_grad()
    torch_no_profiling()
    model = torch_load_jit(model_path, device=device)
    return model


def load_model_onnx2pytorch(model_path, device=None):
    import onnx
    from onnx2pytorch import ConvertModel
    onnx_model = onnx.load(model_path)
    model = ConvertModel(onnx_model)
    print(model)
    from unicon.utils.torch import torch_no_grad, torch_no_profiling
    torch_no_grad()
    torch_no_profiling()
    return model


def load_model_ort(model_path, device=None):
    import onnxruntime as ort
    import torch
    options = ort.SessionOptions()
    # options.enable_profiling = True
    providers = ['CPUExecutionProvider']
    ort_sess = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    inputs = ort_sess.get_inputs()
    input0 = inputs[0]
    print('input0', input0.name, input0.type, input0.shape)
    outputs = ort_sess.get_outputs()
    output0 = outputs[0]
    print('output0', output0.name, output0.type, output0.shape)

    output_names = [x.name for x in outputs]
    input_names = [x.name for x in inputs]

    def model(*args):
        args = [x.numpy() for x in args]
        outputs = ort_sess.run(output_names=output_names, input_feed={k: a for k, a in zip(input_names, args)})
        outputs = [torch.from_numpy(x) for x in outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    return model


def load_model_torch2ort(model_path, in_size=None, **kwds):
    import torch
    jit_model = torch.jit.load(model_path)
    jit_model.eval()
    print(jit_model)
    onnx_path = model_path + '.onnx'
    # if not os.path.exists(onnx_path):
    if True:
        # in_size = [1, 5, 113]
        if in_size is None:
            for ndim in range(1, 5):
                in_size = [100] * ndim
                print('trying', ndim, in_size)
                try:
                    jit_model(torch.randn(*in_size))
                    print('ok', ndim)
                except Exception as e:
                    print('failed', e)
        dummy_input = torch.zeros(*in_size)
        torch.onnx.export(
            jit_model,
            dummy_input,
            onnx_path,
            # export_params=True,
            # opset_version=17,
            # do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    return load_model_ort(onnx_path, **kwds)


_model_type_ext = {
    '.pt': 'torch',
    '.onnx': 'ort',
    # '.onnx': 'onnx2pytorch',
    # '.pt': 'torch2ort',
}

from unicon.utils import import_obj


def load_model(model_path, model_type=None, **kwds):
    if model_type is None:
        _, ext = os.path.splitext(model_path)
        model_type = _model_type_ext[ext]
    loader = import_obj(model_type, default_name_prefix='load_model', default_mod_prefix='unicon.models')
    return loader(model_path, **kwds)
