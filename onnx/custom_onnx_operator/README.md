## 自定义算子流程

1. 在 PyTorch 中实现并注册自定义算子
   - 用C++实现算子逻辑
   - 注册算子`torch::RegisterOperators`
   - 编译为pytorch可加载的`so`，用`python setup.py develop` 进行编译
2. 将 PyTorch 算子导出为 ONNX
   - 注册onnx符号函数 `register_custom_op_symbolic`
   - 导出为onnx模型
3. 在onnx runtime中注册并执行自定义算子
   - 用C++实现onnx runtime版本的算子
   - 编译为`so`
   - 加载推理