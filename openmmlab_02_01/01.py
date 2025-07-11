import torch
from torchvision.models import resnet18

# 使用PyTorch model zoo中的resnet18作为例子
model = resnet18()
model.eval()

# 通过trace的方法生成IR需要一个输入样例
dummy_input = torch.rand(1, 3, 224, 224)

# IR生成
with torch.no_grad():
    # 使用了 trace 模式来生成 IR，所谓 trace 指的是进行一次模型推理，在推理的过程中记录所有经过的计算，将这些记录整合成计算图
    jit_model = torch.jit.trace(model, dummy_input)

jit_layer1 = jit_model.layer1
# TorchScript 的 IR 是可以还原成 python 代码
# print(jit_layer1.code)

# 两种生成IR的模式：trace、script

# 调用inline pass，对graph做变换
torch._C._jit_pass_inline(jit_layer1.graph)
print(jit_layer1.code)

# torch.onnx.export函数可以帮助我们把 PyTorch 模型转换成 ONNX 模型，这个函数会使用 trace 的方式记录 PyTorch 的推理过程