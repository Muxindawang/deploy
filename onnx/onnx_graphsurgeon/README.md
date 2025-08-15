ONNX GraphSurgeon (ONNX-GS) 是一个用于操作和修改 ONNX 模型图的 Python 库，允许开发者在 ONNX 模型的图结构中进行修改、优化、插入节点、删除节点等操作

- **Tensor（张量）**：表示数据的输入、输出或中间结果，分为 `Variable` 和 `Constant`
  - **Variable**：在推理时其值未知的张量。
  - **Constant**：值已知且固定的张量。
- **Node**
- **Graph**



| 常用API                |                                                              |
| ---------------------- | ------------------------------------------------------------ |
| 模型导入               | graph = gs.import_onnx(onnx.load("model.onnx"))              |
| 模型导出               | modified_model = gs.export_onnx(graph)<br /> onnx.save(modified_model, "modified_model.onnx") |
| 创建节点               | gs.Node(op="Relu", name="NewReluNode", inputs=[input_tensor], outputs=[output_tensor]) |
| 创建graph              | gs.Graph(nodes=[node], inputs=[X], outputs=[Y])              |
| 删除节点               | graph.nodes = [node for node in graph.nodes if node.name != "NodeToRemove"] |
| 常量折叠               | graph.fold_constants()                                       |
| 清理未使用的节点和张量 | graph.cleanup()                                              |
| 获取张量               | graph.tensors()                                              |
| 创建常量张量           | gs.Constant(name="W", values=np.ones(shape=(5, 3, 3, 3), dtype=np.float32)) |
| 创建未知张量           | gs.Variable(name="Y", dtype=np.float32, shape=(1, 5, 222, 222)) |
| 注册函数               | @gs.Graph.register() <br />def add(self, a, b):<br />        return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"]) |
| 拓扑排序               | graph.toposort(recurse_subgraphs=False)                      |





### 参考

https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon/examples