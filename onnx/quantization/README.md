## 量化概述

在 ONNX Runtime 中，量化指的是对 ONNX 模型进行 8 位线性量化。

在量化过程中，浮点数被映射到一个 8 位的量化空间，形式如下： `val_fp32 = scale * (val_quantized - zero_point)`

`scale` 是一个正实数，用于将浮点数映射到量化空间。它按以下方式计算：

对于非对称量化：

```
 scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)
```



对于对称量化：

```
scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)
```

`zero_point` 代表量化空间中的零。确保浮点零值在量化空间中能被精确表示非常重要。这是因为许多 CNN 中使用零填充。如果量化后无法唯一表示 0，将导致精度错误。

## ONNX 量化表示格式

有两种方式来表示量化后的 ONNX 模型：

- 面向算子Operator-oriented (QOperator) :

​	所有量化的算子都有自己的 ONNX 定义，例如 QLinearConv、MatMulInteger 等。

​	**每一个算子**（更细一点：权重张量的**每一行/每一列**）单独存一组 scale/zp。

​	**优点**：数值范围更精细，**精度损失小**。**缺点**：scale/zp 数量变多，**内存/带宽微增**。

​	**ONNX 关键词**：`per_channel=True`，例如 Conv/MatMul 的权重。

- 面向张量Tensor-oriented (QDQ; Quantize and DeQuantize) :

​	这种格式在原始算子之间插入 DeQuantizeLinear(QuantizeLinear(tensor))来模拟量化和反量化过程。在静态量化中，QuantizeLinear 和 DeQuantizeLinear 算子也携带量化参数。在动态量化中，会插入 ComputeQuantizationParameters 函数原型来动态计算量化参数。

​	**整个张量**共享同一个 scale/zp。

​	**优点**：只有 1 组 scale/zp，**存储最小**，kernel 实现最简单。**缺点**：不同通道数值差异大时，精度可能下降。

​	**ONNX 默认行为**：`per_channel=False`。

Python API 用于静态量化位于模块 `onnxruntime.quantization.quantize` ，函数 `quantize_static()` 。





动态量化与静态量化的主要区别在于激活值的尺度和零点的计算方式。对于静态量化，这些值是在预先（离线）使用校准数据集计算得出的。因此，激活值在每次前向传递时都具有相同的尺度和零点。而对于动态量化，这些值是在实时（在线）计算，并且针对每次前向传递都是特定的。因此，动态量化更为精确，但会引入额外的计算开销。



## 参考

https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu