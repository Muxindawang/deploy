import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)

    def forward(self, x):
        return self.conv2(x, self.conv1(x))


from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

# 装饰器 @parse_args TorchScript 算子的符号函数要求标注出每一个输入参数的类型。
@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")

def symbolic(g,
             input,
             weight,
             offset,
             mask,
             bias,
             stride_h, stride_w,
             pad_h, pad_w,
             dil_h, dil_w,
             n_weight_grps,
             n_offset_grps,
             use_mask):
    return g.op("custom::deform_conv2d", input, offset)

# 注册符号函数
register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, 'dcn.onnx')
