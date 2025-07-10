from typing import Any

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        # 使用 torch.Tensor.item() 来把只有一个元素的 torch.Tensor 转换成数值
        x = interpolate(x,
            scale_factor=upscale_factor.item(),
            mode='bicubic',
            align_corners=False)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

def init_torch_model():
    torch_model = SuperResolutionNet()

    # 加载权重文件
    # "state_dict" 里面存储模型的权重文件
    state_dict = torch.load('srcnn.pth')['state_dict']

    # 适配权重字典的key（去掉前缀）
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()  # 设置为推理模式
    return torch_model

model = init_torch_model()
input_image = cv2.imread("face.png").astype(np.float32)

input_image = np.transpose(input_image, [2, 0, 1])
input_image = np.expand_dims(input_image, 0)

torch_output = model(torch.from_numpy(input_image), torch.tensor(3)).detach().numpy()

torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

cv2.imwrite("face_torch_2.png", torch_output)

x = torch.randn(1, 3, 256, 256)
# failed
# with torch.no_grad():
#     torch.onnx.export(model, (x, 3),
#                       "srcnn2.onnx",
#                       opset_version=11,
#                       input_names=['input', 'factor'],
#                       output_names=['output'])

# 自定义算子
# 模型的输入参数的类型必须全是torch.Tensor
with torch.no_grad():
    torch.onnx.export(model, (x, torch.tensor(3)),
                      "srcnn2.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])

