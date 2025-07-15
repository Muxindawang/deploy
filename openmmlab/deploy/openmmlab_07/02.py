import torch
import onnx
import tensorrt as trt
from torch.backends.quantized import engine

onnx_model = 'model.onnx'

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

device = torch.device('cuda:0')

torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model, input_names=['input'], output_names=['output'], opset_version=11)
onnx_model = onnx.load(onnx_model)

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

net_work = builder.create_network(EXPLICIT_BATCH)

parser = trt.OnnxParser(net_work, logger)

if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

config = builder.create_builder_config()
# 设置最大工作空间为1GB
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

profile = builder.create_optimization_profile()
profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224])

config.add_optimization_profile(profile)

with torch.cuda.device(device):
    engine = builder.build_serialized_network(net_work, config)
with open('model.engine', mode='wb') as f:
    f.write(bytearray(engine))
    print("generating file done!")