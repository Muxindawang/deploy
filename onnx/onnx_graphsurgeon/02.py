import onnx_graphsurgeon as gs
import numpy as np
import onnx

shape = (1, 3, 224, 224)

x0 = gs.Variable(name="x0", dtype=np.float32, shape=shape)
x1 = gs.Variable(name="x1", dtype=np.float32, shape=shape)

a = gs.Constant("a", values=np.ones(shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape, dtype=np.float32))
mul_out = gs.Variable(name="mul_out")
add_out = gs.Variable(name="add_out")

Y = gs.Variable(name="Y", dtype=np.float32, shape=shape)

nodes = [
    gs.Node(op="Mul", inputs=[a, x1], outputs=[mul_out]),
    gs.Node(op="Add", inputs=[mul_out, b], outputs=[add_out]),
    gs.Node(op="Add", inputs=[add_out, x0], outputs=[Y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x0, x1], outputs=[Y])
model = gs.export_onnx(graph)
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")


model = onnx.load("model.onnx")
graph = gs.import_onnx(model)
tensors = graph.tensors()
graph.inputs = [tensors["x1"].to_variable(dtype=np.float32)]
graph.outputs = [tensors["add_out"].to_variable(dtype=np.float32)]

graph.cleanup()
onnx.save(gs.export_onnx(graph), "subgraph.onnx")
