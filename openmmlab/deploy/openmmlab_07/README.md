## 使用python api构建

首先是使用 Python API 直接搭建 TensorRT 网络，这种方法主要是利用 `tensorrt.Builder` 的 `create_builder_config` 和 `create_network` 功能，分别构建 config 和 network，前者用于设置网络的最大工作空间等参数，后者就是网络主体，需要对其逐层添加内容。

此外，需要定义好输入和输出名称，将构建好的网络序列化，保存成本地文件。值得注意的是：如果想要网络接受不同分辨率的输入输出，需要使用 `tensorrt.Builder` 的 `create_optimization_profile` 函数，并设置最小、最大的尺寸。





## 问题

### onnx->trt的流程是？

1. 准备 ONNX 文件
   - 可用 PyTorch / TF / Paddle / ONNX 本身导出。
   - 推荐用 `onnxsim` 简化模型、`onnx_graphsurgeon` 做图优化。
2. 创建 Logger → Builder → Network → Parser
   - Logger 控制日志级别。
   - Builder 负责统筹资源。
   - Network 承载计算图。
   - Parser 把 ONNX 节点映射到 TensorRT layer。
3. 配置 BuilderConfig
   - 设置 workspace、precision（FP32/FP16/INT8）、算法选择策略、DLA 等。
   - 添加 OptimizationProfile（动态 shape 模型必须）。
   - 若做 INT8 量化，需准备校准数据集并注册 calibrator。
4. 调用 `builder.build_serialized_network()` 或 `build_engine()`
   - 生成序列化的 `.plan` / `.engine` 文件（可再保存到磁盘）。
5. （可选）保存 engine 供后续复用
   - 反序列化时只需 `runtime.deserialize_cuda_engine()`，速度远快于重新 build。

### trt的推理流程是？

- 反序列化 engine

- 创建 ExecutionContext

- 准备 GPU buffer

  - 通过 `engine.get_binding_shape()` / `get_binding_dtype()` 拿到输入输出维度与数据类型。
  - 用 PyTorch (`torch.empty(...).cuda()`)、NumPy + pycuda 或 TensorRT 自带的 `cudaMalloc` 分配显存。

  - 动态 shape 模型需在推理前 `context.set_binding_shape(idx, real_shape)`。

- 推理

  - 同步：`context.execute_v2(bindings)`

  - 异步：`context.execute_async_v2(bindings, stream.handle)``
    - ``bindings` 是 `int` 指针列表，可用 `int(tensor.data_ptr())` 获得。

- 取出输出并后处理

  - 将输出 GPU buffer 拷回 CPU或直接在 GPU 上做后处理。

  - 若输出是 NCHW，注意与框架的 NHWC 区别。

- 清理
  - `cudaStreamDestroy`、`cudaFree` 或让 PyTorch 自动回收张量。



## 参考

https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/06_introduction_to_tensorrt.md

自定义TRT算子：

https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/tutorial/07_write_a_plugin.md