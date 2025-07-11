import onnxruntime_extensions
import openmmlab_04.custom_add  # 注册自定义算子

import torch
import torch.nn as nn
from torch.autograd import Function

class CustomAdd(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y  # 自定义前向逻辑

    @staticmethod
    def symbolic(g, x, y):
        # 用 PythonOp 导出
        return g.op("com.microsoft::PythonOp", x, y, domain_s="ai.onnx.contrib", name_s="custom_add")

class MyModel(nn.Module):
    def forward(self, x, y):
        return CustomAdd.apply(x, y)

model = MyModel()
x = torch.randn(1, 3, 224, 224)
y = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    (x, y),
    "custom_model.onnx",
    opset_version=16,
    input_names=["x", "y"],
    output_names=["output"],
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
)

import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import get_library_path
so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())

sess = ort.InferenceSession("custom_model.onnx", so, providers=["CPUExecutionProvider"])
result = sess.run(None, {
    "x": x.numpy(),
    "y": y.numpy()
})
print(result[0])
