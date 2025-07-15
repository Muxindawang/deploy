## 初识onnx

- 模型部署，指把训练好的模型在特定环境中运行的过程。模型部署要解决模型框架兼容性差和模型运行速度慢这两大问题。
- 模型部署的常见流水线是“深度学习框架-中间表示-推理引擎”。其中比较常用的一个中间表示是 ONNX。
- 深度学习模型实际上就是一个计算图。模型部署时通常把模型转换成**静态**的计算图，即没有控制流（分支语句、循环语句）的计算图。
- PyTorch 框架自带对 ONNX 的支持，只需要构造一组随机的输入，并对模型调用 **torch.onnx.export** 即可完成 PyTorch 到 ONNX 的转换。
- 推理引擎 ONNX Runtime 对 ONNX 模型有原生的支持。给定一个 .onnx 文件，只需要简单使用 ONNX Runtime 的 Python API 就可以完成模型推理。

## 部署中的难题

- 模型动态化。出于性能的考虑，各推理框架都默认模型的输入形状、输出形状、结构是静态的。而为了让模型的泛用性更强，部署时，需要在尽可能不影响原有逻辑的前提下，让模型的输入输出或者结构**动态化**。
- 新算子的实现。深度学习技术日新月异，提出新算子的速度往往快于 ONNX 维护者支持的速度。为了部署最新的模型，部署工程师往往需要自己在 ONNX 和推理引擎中支持新算子。
- 中间表示与推理引擎的兼容问题。由于各推理引擎的实现不同，对 ONNX 难以形成统一的支持。为了确保模型在不同的推理引擎中有同样的运行效果，部署工程师往往得为某个推理引擎定制模型代码，这为模型部署引入了许多工作量。



- 模型部署中常见的几类困难有：**模型的动态化；新算子的实现；框架间的兼容**。
- PyTorch 转 ONNX，实际上就是把每一个操作转化成 ONNX 定义的某一个算子。比如对于 PyTorch 中的 Upsample 和 interpolate，在转 ONNX 后最终都会成为 ONNX 的 Resize 算子。
- 通过修改继承自 torch.autograd.Function 的算子的 symbolic 方法，可以改变该算子映射到 ONNX 算子的行为。

## 初识TorchScript

TorchScript工具 ：包括代码的追踪及解析、中间表示的生成、模型优化、序列化等各种功能，可以说是覆盖了模型部署的方方面面。

- 模型转换
  -  trace 模式：进行一次模型推理，在推理的过程中记录所有经过的计算，将这些记录整合成计算图。
  - script模式：直接解析网络定义的 python 代码，生成抽象语法树 AST，因此这种方法可以解决一些 trace 无法解决的问题，比如对 branch/loop 等数据流控制语句的建图。
- 模型优化
- 序列化

`torch.onnx.export`函数可以帮助我们把 PyTorch 模型转换成 ONNX 模型，这个函数会使用 trace 的方式记录 PyTorch 的推理过程。

1. 使用 trace 的方式先生成一个 TorchScipt 模型，如果你转换的本身就是 TorchScript 模型，则可以跳过这一步。
2. 使用许多 pass 对 1 中生成的模型进行变换，其中对 ONNX 导出最重要的一个 pass 就是`ToONNX`，这个 pass 会进行一个映射，将 TorchScript 中`prim`、`aten`空间下的算子映射到`onnx`空间下的算子。
3. 使用 ONNX 的 proto 格式对模型进行序列化，完成 ONNX 的导出。

## 自定义开发



要使 PyTorch 算子顺利转换到 ONNX ，我们需要保证以下三个环节都不出错：

- 算子在 PyTorch 中有实现
  - 组合现有算子
  - 添加 TorchScript 算子
  - 添加普通 C++ 拓展算子
- 有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法
  - 为 ATen 算子添加符号函数
  - 为 TorchScript 算子添加符号函数
  - 封装成 `torch.autograd.Function` 并添加符号函数
- ONNX 有相应的算子
  - 使用现有 ONNX 算子
  - 定义新 ONNX 算子

情况一：aten中有实现，缺少映射到onnx算子的符号函数，此时需要补充符号函数

1. 获取aten中算子的接口定义

2. 添加符号函数，所谓的符号函数，可以看成是PyTorch算子类的一个静态方法，在PyTorch模型转onnx模型时，各个PyTorch算子的符号函数会被依次调用，以完成 PyTorch 算子到 ONNX 算子的转换。

   ```python
   def symbolic(g: torch._C.Graph, input_0: torch._C.Value, input_1: torch._C.Value, ...): 
       
   # 第一个参数就固定叫 g，它表示和计算图相关的内容；后面的每个参数都表示算子的输入，需要和算子的前向推理接口的输入相同。
   # g 有一个方法 op。在把 PyTorch 算子转换成 ONNX 算子时，需要在符号函数中调用此方法来为最终的计算图添加一个 ONNX 算子。
   
   def op(name: str, input_0: torch._C.Value, input_1: torch._C.Value, ...) 
   ```





| 步骤           | 方法                     | 工具                                           |
| -------------- | ------------------------ | ---------------------------------------------- |
| 自定义算子实现 | Python（Function）或 C++ | `torch.autograd.Function` / `TORCH_LIBRARY`    |
| 注册算子       | Python 注册符号函数      | `register_custom_op_symbolic`                  |
| 导出 ONNX      | `torch.onnx.export`      | 设置 `opset_version` 和 `operator_export_type` |
| 推理验证       | ONNX Runtime（可选）     | 需实现自定义算子支持                           |





## onnx模型调试修改

- ONNX 使用 Protobuf 定义规范和序列化模型。
- 一个 ONNX 模型主要由 `ModelProto`,`GraphProto`,`NodeProto`,`ValueInfoProto` 这几个数据类的对象组成。
- 使用 `onnx.helper.make_xxx`，我们可以构造 ONNX 模型的数据对象。
- `onnx.save()` 可以保存模型，`onnx.load()` 可以读取模型，`onnx.checker.check_model()` 可以检查模型是否符合规范。
- `onnx.utils.extract_model()` 可以从原模型中取出部分节点，和新定义的输入、输出边构成一个新的子模型。
- 利用子模型提取功能，我们可以输出原 ONNX 模型的中间结果，实现对 ONNX 模型的调试。
- 模型裁剪时，需要保证子模型给定的输入包含下面节点所需的全部输入，如下图，如果裁切的子模型中给定输入是3的输入，输出是5的输出，则会报错，因为缺少节点2、4这一侧的输入

![image-20250711103638634](/home/ll/.config/Typora/typora-user-images/image-20250711103638634.png)





## 问题

1. with torch,no_grad的作用？
2. 导出onnx之后怎么对比精度？什么方法对比？
3. onnx模型都包含哪些属性？表示什么含义？
