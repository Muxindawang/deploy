

### 基本概念

- 模型
  - graph：模型的核心部分，定义了模型的计算流程。
    - graph由nodes、tensors、initilizers、inputs、outputs组成
  - tensors：数据的载体，可以是输入、输出或中间结果。
    - 变量
    - 常量
  - nodes：表示图中的操作（如卷积、激活函数等）。
    - Optype：操作类型，如conv、ReLU
    - name
    - inputs、outputs
    - attributes：额外属性，如conv的stride
  - initializers：模型的参数，通常是训练好的权重和偏置

### onnx格式-proto

https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

https://onnx.ai/onnx/api/classes.html#graphproto

```protobuf
message ModelProto {
  optional int64 ir_version = 1;
  repeated OperatorSetIdProto opset_import = 8;
  optional string producer_name = 2;
  optional string producer_version = 3;
  optional string domain = 4;
  optional int64 model_version = 5;
  optional string doc_string = 6;
  optional GraphProto graph = 7;    // The parameterized graph that is evaluated to execute the model.
  repeated StringStringEntryProto metadata_props = 14;
  repeated TrainingInfoProto training_info = 20;
  repeated FunctionProto functions = 25;
  repeated DeviceConfigurationProto configuration = 26;
};

message GraphProto {
  // The nodes in the graph, sorted topologically.
  repeated NodeProto node = 1;

  // The name of the graph.
  optional string name = 2;   // namespace Graph

  // A list of named tensor values, used to specify constant inputs of the graph.
  // Each initializer (both TensorProto as well SparseTensorProto) MUST have a name.
  // The name MUST be unique across both initializer and sparse_initializer,
  // but the name MAY also appear in the input list.
  repeated TensorProto initializer = 5;
  repeated SparseTensorProto sparse_initializer = 15;
  optional string doc_string = 10;
  repeated ValueInfoProto input = 11;
  repeated ValueInfoProto output = 12;
  repeated ValueInfoProto value_info = 13;
  repeated TensorAnnotation quantization_annotation = 14;
  repeated StringStringEntryProto metadata_props = 16;
  reserved 3, 4, 6 to 9;
  reserved "ir_version", "producer_version", "producer_tag", "domain";
}

message NodeProto {
  repeated string input = 1;    // namespace Value
  repeated string output = 2;   // namespace Value

  // An optional identifier for this node in a graph.
  // This field MAY be absent in this version of the IR.
  optional string name = 3;     // namespace Node

  // The symbolic identifier of the Operator to execute.
  optional string op_type = 4;  // namespace Operator
  // The domain of the OperatorSet that specifies the operator named by op_type.
  optional string domain = 7;   // namespace Domain
  // Overload identifier, used only to map this to a model-local function.
  optional string overload = 8;

  // Additional named attributes.
  repeated AttributeProto attribute = 5;

  // A human-readable documentation for this node. Markdown is allowed.
  optional string doc_string = 6;

  // Named metadata values; keys should be distinct.
  repeated StringStringEntryProto metadata_props = 9;

  // Configuration of multi-device annotations.
  repeated NodeDeviceConfigurationProto device_configurations = 10;
}
```



### 算子

算子列表：https://onnx.ai/onnx/operators/index.html

### 自定义算子

https://github.com/onnx/tutorials/blob/main/PyTorchCustomOperator/README.md

### 图优化

https://github.com/onnx/optimizer

### 量化

参考：https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization

### 常用API

| 功能分类         | 功能描述      | API 调用                            | 示例代码                                                     |
| :--------------- | :------------ | :---------------------------------- | :----------------------------------------------------------- |
| **ONNX**         | 加载模型      | `onnx.load`                         | `python model = onnx.load("model.onnx") `                    |
|                  | 保存模型      | `onnx.save`                         | `python onnx.save(model, "model.onnx") `                     |
|                  | 验证模型      | `onnx.checker.check_model`          | `python onnx.checker.check_model(model) `                    |
|                  | 打印模型结构  | `onnx.helper.printable_graph`       | `python print(onnx.helper.printable_graph(model.graph)) `    |
|                  | 形状推理      | `onnx.shape_inference.infer_shapes` | `python inferred_model = onnx.shape_inference.infer_shapes(model) ` |
|                  | 提取子模型    | `onnx.utils.extract_model`          | `python extract_model(input_path, output_path, input_names, output_names) ` |
|                  | Pytorch转onnx | torch.onnx.export                   | torch.onnx.export(model, dummy_input, "model.onnx", export_params=True, opset_version=11, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}) |
| 图优化           |               | onnxoptimizer                       |                                                              |
| 量化             |               | onnxruntime.quantization            |                                                              |
| **ONNX Runtime** | 创建推理会话  | `ort.InferenceSession`              | `python session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"]) ` |
|                  | 获取输入信息  | `session.get_inputs()`              | `python input_name = session.get_inputs()[0].name `          |
|                  | 获取输出信息  | `session.get_outputs()`             | `python output_name = session.get_outputs()[0].name `        |
|                  | 执行推理      | `session.run`                       | `python outputs = session.run([output_name], {input_name: input_data}) ` |



onnx官方文档：https://onnx.ai/onnx/

算子列表：https://onnx.ai/onnx/operators/index.html

