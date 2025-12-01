import os
import torch


class RKNNModel():

    def __init__(self, model_path, rebuild=False, export=True, target_platform='rk3588'):
        try:
            from rknn.api import RKNN
        except ImportError as e:
            print(e)
            RKNN = None
        try:
            from rknnlite.api import RKNNLite
        except ImportError as e:
            print(e)
            RKNNLite = None
        rknn = RKNN(verbose=True)
        self.rknn = rknn

        rknn.config(target_platform=target_platform)

        rknn_path = model_path.replace('.onnx', '.rknn')
        if os.path.exists(rknn_path) and not rebuild:
            print('loading rknn', rknn_path)
            # rknn_lite = RKNNLite()
            ret = rknn.load_rknn(rknn_path)
        else:
            print('loading onnx', model_path)
            ret = rknn.load_onnx(model=model_path)
            if ret != 0:
                print('Load model failed!')
                return

            print('building model')
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                print('Build model failed!')
                exit(ret)

            if export:
                print('export rknn model')
                ret = rknn.export_rknn(rknn_path)
                if ret != 0:
                    print('Export rknn model failed!')
                    exit(ret)

        ret = rknn.init_runtime(target=target_platform)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)

    def __call__(self, *args):
        args = [x.numpy() for x in args]
        outputs = self.rknn.inference(inputs=args)
        outputs = [torch.from_numpy(x) for x in outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    def __del__(self):
        self.rknn.release()


def load_model_rknn(model_path, device=None, **kwds):
    return RKNNModel(model_path, **kwds)
