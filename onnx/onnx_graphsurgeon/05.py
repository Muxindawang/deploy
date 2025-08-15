import onnx_graphsurgeon as gs
import onnx
import numpy as np

shape = (1, 3)

input = gs.Variable("input", dtype=np.float32, shape=shape)

a = gs.Constant(name="a", values=np.ones(shape, dtype=np.float32))
b = gs.Constant(name="b", values=np.ones(shape, dtype=np.float32))
c = gs.Variable(name="c")
d = gs.Constant(name="d", values=np.ones(shape, dtype=np.float32))
e = gs.Variable(name="e")

output = gs.Variable("output", shape=shape, dtype=np.float32)

nodes = [
    gs.Node(op="Add", inputs=[a, b], outputs=[c]),
    gs.Node(op="Add", inputs=[c, d], outputs=[e]),
    gs.Node(op="Add", inputs=[input, e], outputs=[output]),
]

graph = gs.Graph(nodes, inputs=[input], outputs=[output])
model = gs.export_onnx(graph)
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")

graph = gs.import_onnx(onnx.load("model.onnx"))
# 使用ONNX Runtime在图中折叠常量。这将用常量张量替换可以在运行时之前评估的表达式。
# fold_constants() 函数不会移除它替换掉的节点——它只是简单地更改后续节点的输入。
# 为了移除这些未使用的节点，我们可以在调用 fold_constants() 之后使用 cleanup()。

# fold_constants()
# 折叠常量（Constant Folding）。这个操作会查找模型中的常量节点，并尝试将它们的值提前计算出来。
# 在模型中，有些节点的输入是常量，而这些节点的输出也可以在模型运行之前就计算出来。例如，一个简单的加法节点，如果它的两个输入都是常量，那么它的输出也是一个常量。fold_constants() 会将这些可以提前计算的节点的输出直接替换为常量张量。
# 通过折叠常量，可以减少模型在运行时的计算量，提高模型的推理速度。

graph.fold_constants().cleanup()
onnx.save(gs.export_onnx(graph), "folded.onnx")