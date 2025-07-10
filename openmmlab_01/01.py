import os
import cv2
import numpy as np
import requests
import torch
import torch.onnx
from torch import nn
import onnx

# 定义超分辨率神经网络模型
class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        # 使用双三次插值进行上采样
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out
# 下载模型权重和测试图片，如果本地不存在的话
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
        'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        # 下载文件并保存到本地
        open(name, 'wb').write(requests.get(url).content)

# 初始化PyTorch模型，加载权重

def init_torch_model():
    torch_model = SuperResolutionNet(upscale_factor=3)

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

# 初始化模型
model = init_torch_model()
# 读取测试图片
input_img = cv2.imread('face.png').astype(np.float32)

# HWC（高宽通道）转为NCHW（批次通道高宽）
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# 推理，得到超分辨率结果
# .detach() - 分离计算图，切断梯度
torch_output = model(torch.from_numpy(input_img)).detach().numpy()

# NCHW转回HWC
# 移除张量的第一个维度（批次维度）
torch_output = np.squeeze(torch_output, 0)
# 将像素值限制在 0-255 范围内
torch_output = np.clip(torch_output, 0, 255)
# 转换维度顺序并改变数据类型
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# 保存结果图片
cv2.imwrite("face_torch.png", torch_output)

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "srcnn.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])

onnx_model = onnx.load("srcnn.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")

import onnxruntime
# 创建 ONNX Runtime 推理会话
ort_session = onnxruntime.InferenceSession("srcnn.onnx")

ort_inputs = {'input': input_img}
# 执行模型推理
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_ort.png", ort_output)