# SPDX-License-Identifier: Apache-2.0

import torch


def register_custom_op():
    def my_group_norm(g, input, num_groups, scale, bias, eps):
        return g.op("mydomain::testgroupnorm", input, num_groups, scale, bias, epsilon_f=0.)

    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("mynamespace::custom_group_norm", my_group_norm, 9)


def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, x, num_groups, scale, bias):
            return torch.ops.mynamespace.custom_group_norm(x, num_groups, scale, bias, 0.)

    X = torch.randn(3, 2, 1, 2)
    num_groups = torch.tensor([2.])
    scale = torch.tensor([1., 1.])
    bias = torch.tensor([0., 0.])
    inputs = (X, num_groups, scale, bias)

    f = './model.onnx'
    torch.onnx.export(CustomModel(), inputs, f,
                      opset_version=9,
                      example_outputs=None,
                      input_names=["X", "num_groups", "scale", "bias"], output_names=["Y"],
                      custom_opsets={"mydomain": 1})


torch.ops.load_library(
    "build/lib.linux-x86_64-3.11/custom_group_norm.cpython-311-x86_64-linux-gnu.so")
register_custom_op()
export_custom_op()

## 由于onnxruntime缺少这个算子的C++实现，所以需要onnxruntime自定义实现之后，才可以进行推理
# import onnxruntime as ort
# from onnxruntime_extensions import get_library_path
#
# so = ort.SessionOptions()
# so.register_custom_ops_library(get_library_path())
# sess = ort.InferenceSession('model.onnx', so, )
# X = torch.randn(3, 2, 1, 2)
# num_groups = torch.tensor([2.])
# scale = torch.tensor([1., 1.])
# bias = torch.tensor([0., 0.])
# inputs = (X, num_groups, scale, bias)
# result = sess.run(None, inputs)
# print(result)