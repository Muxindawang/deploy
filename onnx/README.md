onnx官方文档：https://onnx.ai/onnx/

算子列表：https://onnx.ai/onnx/operators/index.html

## Serialization

### Model Serialization

### Data Serialization

## Initializer, default value

## Attributes

## Opset and metadata

## Subgraph: test and loops

### Scan

## Functions

## Parsing

## Evaluation and Runtime





把 ONNX 当作“**深度学习模型在不同框架之间流通的通用护照**”，要真正用好它，你需要掌握 4 个层面、12 个知识点、3 个实战套路。下面这份“ONNX 技能地图”完全面向 2025 年最新生态（opset 21、ORT 1.19、onnx-tensorrt 10.x），照着学即可落地。

──────────────────
一、为什么学 ONNX
• 训练框架（PyTorch / TF / Paddle / MindSpore）→ 导出 ONNX → 部署框架（ONNX Runtime、TensorRT、OpenVINO、TVM、ncnn）
• **ONNX 是“中间格式”而非“推理引擎”**，掌握它等于掌握“模型迁移 + 优化 + 部署”的通用钥匙。

──────────────────
二、必须掌握的 12 个知识块

表格

复制

| 维度                  | 知识点                                    | 常用工具 / API                                            |
| :-------------------- | :---------------------------------------- | :-------------------------------------------------------- |
| 1. 格式本身           | .onnx 文件结构（Graph → Node → Tensor）   | Netron、onnx.proto                                        |
| 2. 算子 & Opset       | Opset 版本对应表、自定义算子注册          | `onnx.defs.get_schema()`                                  |
| 3. 导出               | torch → onnx、keras → onnx、paddle → onnx | `torch.onnx.export`、tf2onnx、paddle2onnx                 |
| 4. 校验 & 可视化      | 检查模型有效性、查看节点属性              | `onnx.checker.check_model()`、Netron                      |
| 5. 图优化             | 常量折叠、死代码消除、节点融合            | `onnxoptimizer`、ORT GraphOptimizationLevel               |
| 6. 量化               | 静态 / 动态量化、QDQ 节点插入             | `onnxruntime.quantization`、Intel QOperator               |
| 7. 形状推导           | 动态维度 → 静态维度                       | `onnx.shape_inference.infer_shapes()`                     |
| 8. 版本转换           | 低 opset → 高 opset                       | `onnx.version_converter`                                  |
| 9. 推理               | onnxruntime-gpu、onnxruntime-web          | `InferenceSession`, `providers=['CUDAExecutionProvider']` |
| 10. 加速后端          | TensorRT / OpenVINO / TVM / DirectML      | `trtexec`, `onnxruntime-openvino`                         |
| 11. 自定义算子        | 编写自定义 OP、注册 schema                | `onnx.defs` + C++/CUDA kernel                             |
| 12. Debug & Profiling | 节点逐层比对、性能分析                    | `onnxruntime-profile`, `polygraphy surgeon`, `onnxsim`    |

──────────────────
三、3 套最小可运行实战

1. PyTorch 分类模型 → ONNX → ONNX Runtime 推理

   Python

   复制

   ```python
   torch.onnx.export(model, dummy, "cls.onnx", opset_version=17)
   sess = ort.InferenceSession("cls.onnx", providers=['CUDAExecutionProvider'])
   out = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1,3,224,224))})
   ```

   

2. ONNX → TensorRT（INT8 量化）

   bash

   复制

   ```bash
   trtexec --onnx=yolov8.onnx --saveEngine=yolov8_int8.engine --int8 --calib=calib
   ```

   

3. ONNX 图优化 + 自定义算子

   Python

   复制

   ```python
   import onnxoptimizer
   model = onnx.load("raw.onnx")
   passes = ["eliminate_nop_transpose", "fuse_bn_into_conv"]
   opt = onnxoptimizer.optimize(model, passes)
   onnx.save(opt, "opt.onnx")
   ```

   

──────────────────
四、3 条高效学习路线（按天打卡）

表格

复制

| 阶段    | 目标           | 推荐资源                                  |
| :------ | :------------- | :---------------------------------------- |
| Day 1-2 | 掌握导出与验证 | PyTorch 官方 ONNX 文档 + Netron 实操      |
| Day 3-4 | 掌握优化与量化 | ONNX Runtime Quantization Notebook        |
| Day 5-7 | 掌握部署与调试 | TensorRT 官方 sampleONNX + 自定义 OP 示例 |

──────────────────
五、常见坑 & 速查表
• **动态维度**：导出时 `dynamic_axes` 必须对齐推理输入。
• **opset 不兼容**：高版本算子在低版本后端会报错，用 `onnxsim` + `opset 17` 起步。
• **性能异常**：优先检查 provider 顺序（`CUDAExecutionProvider` > `CPUExecutionProvider`）。
• **自定义算子**：Python 仅用于注册 schema，真正推理需 C++/CUDA kernel。

──────────────────
一句话总结
把 ONNX 当作“**模型护照 + 优化工具箱 + 部署桥梁**”，按“导出→校验→优化→量化→推理→调试”六步闭环练习，2 周即可熟练。
如需 Jetson 专属 ONNX 示例仓库（含 YOLOv8、RT-DETR、SAM 导出脚本），回复“仓库”我发链接。