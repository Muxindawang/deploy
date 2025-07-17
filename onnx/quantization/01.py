from torchvision import models, datasets, transforms as T
# 加载预训练的MobileNetV2模型
mobilenet_v2 = models.mobilenet_v2(pretrained=True)

import torch

image_height = 224
image_width = 224
# 构造一个随机输入张量
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
# 用PyTorch模型推理一次，检查模型可用
torch_out = mobilenet_v2(x)

# 导出为ONNX模型
# 注意：intput_names拼写应为input_names，否则会报错
torch.onnx.export(mobilenet_v2,
                  x,
                  "mobilenet_v2_float.onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],  # 正确拼写
                  output_names=['output'])

from PIL import Image
import numpy as np
import onnxruntime

# 图像预处理函数
# 读取图片，resize，归一化，转为NCHW格式
# 注意：mean和std的写法与官方略有不同

def preprocess_image(image_path, height, width, channels=1):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0 ,1])
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel] ) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

# 读取ImageNet类别标签
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# 创建ONNX Runtime推理会话（全精度模型）
session_fp32 = onnxruntime.InferenceSession('mobilenet_v2_float.onnx')

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 单张图片推理并输出Top5类别
def run_sample(session, image_file, categories):
    output = session.run([], {'input':preprocess_image(image_file, image_height, image_width)})[0]
    output = output.flatten()
    output = softmax(output)
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], output[catid])

# 测试全精度模型推理
run_sample(session_fp32, 'cat.jpg', categories)

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import os

# 批量图片预处理函数，用于量化校准
def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

# 量化校准数据读取器
class MobilenetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, image_height, image_width, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

# 指定校准图片文件夹
calibration_data_folder = 'calibration_imagenet'
dr = MobilenetDataReader(calibration_data_folder)

# 静态量化模型，生成uint8模型
quantize_static('mobilenet_v2_float.onnx',
                'mobilenet_v2_uint8.onnx',
                dr)

# 打印模型体积对比
print('ONNX full precision model size (MB):', os.path.getsize("mobilenet_v2_float.onnx")/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize("mobilenet_v2_uint8.onnx")/(1024*1024))

# 加载量化模型并推理
session_quant = onnxruntime.InferenceSession("mobilenet_v2_uint8.onnx")
run_sample(session_quant, 'cat.jpg', categories)
run_sample(session_quant, 'cat.jpg', categories)

