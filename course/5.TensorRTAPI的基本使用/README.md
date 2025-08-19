### TensorRT基本工作流程

1. **export the model**
2. **select a batch size**
3. **select a precision**
4. **convert the model**
5. **deploy the model**



#### TensorRT 模型转换

- 手动使用TensorRT API搭建，类似于pytorch等搭建卷积、ReLU等，并设置必要的layer参数
- onnx模型生成TensorRT引擎（可以使用`trtexec`命令）
- PyTorch模型生成TensorRT引擎

**创建builder、config、network**

```c++
Logger logger;
// 要构建引擎，创建一个 builder，并传递一个为 TensorRT 创建的 logger，该 logger 用于在网络中报告错误、警告和信息消息
nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
const uint32_t explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
```

**手动创建**

```c++
// 添加input
ITensor *data = network->addInput("input", DataType::kFLOAT, Dims4{maxBatchSize, channel, H, W});
// 添加conv 卷积核大小5*5 需要添加权重和偏置
IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 6, DimsHW{5, 5}, weightMap["conv1.weight"], weightMap["conv1.bias"]);
// 设置卷积步长
conv1->setStrideNd(DimsHW{1, 1});
// 添加激活层 ReLU
IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
// 添加linear层
IFullyConnectedLayer* fc = network->addFullyConnected(*relu1->getOutput(0), OUTPUT_SIZE, weightMap["fc.weight"], weightMap["fc.bias"]);
// 添加softmax
ISoftMaxLayer* prob = network->addSoftMax(*fc->getOutput(0));
// 设置输出name
prob->getOutput(0)->setName("ouput");
network->markOutput(*prob->getOutput(0));

// 创建引擎
builder->setMaxBatchSize(maxBatchSize);		// 设置batch大小
config->setMaxWorkspaceSize(16 << 20);    // 16MB
// 构建引擎
ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
// 序列化引擎
IHostMemory* modelStream = engine->serialize();
// 保存引擎文件
std::ofstream p("model.engine", std::ios::binary);
p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

// 释放空间
engine->destroy();   // 销毁引擎
config->destroy();   // 销毁配置
builder->destroy();  // 销毁构建器
modelStream->destroy();  // 销毁stream

```

**onnx转换**

```c++
// 创建parser
auto parser = nvonnxparser::createParser(*network, logger);
// onnx -> network
// bool parseFromFile(const char* onnxModelFile, int verbosity) 
bool parsed = parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO));
```

参考：https://github.com/NVIDIA/TensorRT/tree/v8.6.1/samples/sampleOnnxMNIST





**推理**

```c++
std::vector<char> engineData(fsize);
engineFile.read(engineData.data(), fsize);

nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
// 从文件中反序列化 TensorRT 引擎。文件内容被读入缓冲区并在内存中反序列化。
ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size, nullptr);
// 创建上下文
IExecutionContext* context = engine->createExecutionContext();
// 准备数据
float input_data[B*C*H*C]
float output_data[B*OUTPUT_SIZE]

// 分配GPU内存
void* buffers[2];

// 获取输入输出绑定索引
const int inputIndex = engine.getBindingIndex("input_name");
const int outputIndex = engine.getBindingIndex("output_name");

// 分配设备内存
CHECK(cudaMalloc(&buffers[inputIndex], B * C * H * W * sizeof(float)));
CHECK(cudaMalloc(&buffers[outputIndex], B * OUTPUT_SIZE * sizeof(float)));
// 创建CUDA流
cudaStream_t stream;
// 异步数据传输
cudaMemcpyAsync(buffers[inputIndex], input, 
    B * C * H * W * sizeof(float), cudaMemcpyHostToDevice, stream);
// 推理
context.enqueueV2(buffers, stream, nullptr);
// 传回数据
cudaMemcpyAsync(output, buffers[outputIndex], B * OUTPUT_SIZE * sizeof(float), 
    cudaMemcpyDeviceToHost, stream)；
cudaStreamSynchronize(stream);

// 清理资源
cudaStreamDestroy(stream);
CHECK(cudaFree(buffers[inputIndex]));
CHECK(cudaFree(buffers[outputIndex]));
```



## 参考

https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/quick-start-guide/index.html#ecosystem

https://docs.nvidia.com/deeplearning/tensorrt/archives/