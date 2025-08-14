**TensorRT的工作流程**

- 将训练好的模型转为 TensorRT 可以识别的模型
- TensorRT 逐个 Layer 进行分析并尝试各种优化策略（量化、层融合、调度、多流执行、内存复用等等）
- 生成推理引擎，可以从 C++/Python 程序中调用

部署时一般是 Pytorch->ONNX->TensorRT

**TensorRT的一些限制**

- 针对不支持的算子
  - 看看不同 TensorRT 版本中是否有做支持
  - 修改 Pytorch 的算子选择，让它使用一些 TensorRT 支持的算子
  - 自己写插件，内部实现自定义算子以及自定义 CUDA 加速核函数
  - 不使用 ONNX 自己创建 parser 直接调用 TensorRT API 逐层创建网络
- 不同TensorRT版本的优化策略是不一样的
  - 比如对 Transformer 的优化 TensorRT-7.x 和 TensorRT-8.x 跑出来的性能是不一样的
- 有时你预期TensorRT的优化和实际的优化是不一样的
  - 比如说你期望 TensorRT 使用 Tensor core 但 kernel autotuning 之后 TensorRT 觉得使用 Tensor core 反而会效率降低，结果给你分配 CUDA core 使用
  - 比如说 INT8 量化的结果比 FP16 还要慢，这是 TensorRT 内部的一些优化调度所导致的
- 天生并行性差的layer，TensorRT也没有办法
  - 1x1 conv 这种 layer 再怎么优化也没有 7x7 conv 并行效果好

**TensorRT的优化策略**

- 层融合
  - 垂直层融合，CBL融合
    - 层融合的优点就在于它可以减少启动 kernel 的开销与 memory 操作，从而提高效率，同时有些计算可以通过层融合优化后跟其它计算合并
  - 水平层融合，水平方向同类layer，会直接进行融合
    - 水平层融合在 transformer 和 self-attention 中还是比较常见的，在 transformer 中 q、k、v 的生成过程会用到水平的层融合

- Kernel auto-tuning

  - TensorRT内部对于同一个层使用各种不同kernel函数进行性能测试，比如对于FC层中的矩阵乘法，根据tile size有很多中kernel function， (e.g. 32x32, 32x64, 64x64, 64x128, 128x128，针对不同硬件有不同策略)

- 量化

  - 压缩模型，将FP32压缩为FP16或者INT8

  

**onnx**

- ModelProto：描述的是整个模型的信息

  - GraphProto：描述的是整个网络的信息
    - NodeProto：描述的是各个计算节点，比如 conv，linear
    - TensorProto：描述的是 tensor 的信息，主要包括权重
    - ValueInfoProto：描述的是 input/output 信息

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bcdefab66e6645248bf2d8ed72eb30b0.png#pic_center)