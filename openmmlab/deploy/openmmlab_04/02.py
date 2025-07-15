import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


from torch.onnx.symbolic_registry import register_op

# asinh_symbolic 就是 Asinh 的符号函数 从除了g以外的第二个输入开始都要严格对应它在ATen中的定义
# 在符号函数的函数体中，g.op("Asinh", input)则完成了 ONNX 算子的定义。
def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

# 把这个符号函数和原来的 ATen 算子“绑定”起来
register_op('asinh', asinh_symbolic, '', 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, 'asinh.onnx')