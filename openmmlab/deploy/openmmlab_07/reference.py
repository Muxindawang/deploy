# reference
from typing import Union, Optional, Sequence, Dict, Any

import torch
import tensorrt as trt

class TRTWrapper(torch.nn.Module):
    def __init__(self,
                 engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()

        # 1. 反序列化引擎：如果传入的是文件路径，则加载engine文件
        if isinstance(engine, str):
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(engine, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.engine = engine
        # 创建推理上下文
        self.context = self.engine.create_execution_context()

        # 2. 收集输入/输出名（TensorRT 10推荐API）
        num_tensors = self.engine.num_io_tensors  # 总的输入输出张量数
        names = [self.engine.get_tensor_name(i) for i in range(num_tensors)]  # 所有张量名

        # 获取所有输入名
        self.input_names = [
            n for n in names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        # 获取所有输出名
        self.output_names = (
            list(output_names) if output_names is not None else
            [n for n in names
             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取当前CUDA流
        stream = torch.cuda.current_stream().cuda_stream

        # ---- 1. 设置输入 shape 与地址 ----
        for name in self.input_names:
            tensor = inputs[name].contiguous()  # 保证内存连续
            if tensor.dtype == torch.long:
                tensor = tensor.int()  # TensorRT不支持long，转为int
            self.context.set_input_shape(name, tensor.shape)  # 设置输入shape
            self.context.set_tensor_address(name, tensor.data_ptr())  # 设置输入数据地址

        # ---- 2. 分配并绑定输出 ----
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)  # 获取输出shape
            out_tensor = torch.empty(
                tuple(shape), dtype=torch.float32, device='cuda')  # 分配输出张量
            outputs[name] = out_tensor
            self.context.set_tensor_address(name, out_tensor.data_ptr())  # 设置输出数据地址

        # ---- 3. 执行推理 ----
        ok = self.context.execute_async_v3(stream)  # 异步执行
        if not ok:
            raise RuntimeError("TensorRT execute failed")

        return outputs

# 示例用法：加载engine并推理
model = TRTWrapper('model.engine', ['output'])
output = model(dict(input = torch.randn(1, 3, 224, 224).cuda()))
print(output)