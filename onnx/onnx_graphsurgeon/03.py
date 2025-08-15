import onnx
import onnx_graphsurgeon as gs
import numpy as np

shape = (1, 3, 224, 224)

x0 = gs.Variable(name="x0", dtype=np.float32, shape=shape)
x1 = gs.Variable(name="x1", dtype=np.float32, shape=shape)

a = gs.Constant(name="a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant(name="b", values=np.ones(shape=shape, dtype=np.float32))

mul_out = gs.Variable(name="mul_out")
add_out = gs.Variable(name="add_out")

Y = gs.Variable(name="Y", dtype=np.float32, shape=shape)

nodes = [
    gs.Node(op="Mul", inputs=[a, x1], outputs=[mul_out]),
    gs.Node(op="Add", inputs=[mul_out, b], outputs=[add_out]),
    gs.Node(op="Add", inputs=[x0, add_out], outputs=[Y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x0, x1], outputs=[Y])
model = gs.export_onnx(graph)
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")

graph = gs.import_onnx(onnx.load("model.onnx"))

# 1. Remove the `b` input of the add node
first_add = [node for node in graph.nodes if node.op == "Add"][0]
first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"]

# 2. Change the Add to a LeakyRelu
first_add.op = "LeakyRelu"
first_add.attrs["alpha"] = 0.02
#
# # 3. Add an identity after the add node
identity_out = gs.Variable("identity_out", dtype=np.float32)
identity = gs.Node(op="Identity", inputs=first_add.outputs, outputs=[identity_out])
graph.nodes.append(identity)
# # 4. Modify the graph output to be the identity output
graph.outputs = [identity_out]
graph.cleanup(remove_unused_graph_inputs=True).toposort()

# onnx.shape_inference.infer_shapes 用于自动推断 ONNX 模型中各个张量的形状信息。
model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
onnx.save(model, "modified.onnx")