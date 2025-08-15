import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Inputs
x = gs.Variable(name="x", dtype=np.float32, shape=(1, 3, 224, 224))

# Intermediate tensors
i0 = gs.Variable(name="i0")
i1 = gs.Variable(name="i1")

# Outputs
y = gs.Variable(name="y", dtype=np.float32)

nodes = [
    gs.Node(op="Identity", inputs=[x], outputs=[i0]),
    gs.Node(op="FakeNodeToRemove", inputs=[i0], outputs=[i1]),
    gs.Node(op="Identity", inputs=[i1], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])

model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
onnx.save(model, "model.onnx")

graph = gs.import_onnx(onnx.load("model.onnx"))

fk_node = [nd for nd in graph.nodes if nd.op == "FakeNodeToRemove"][0]
# Node provides i() and o() functions that can optionally be provided an index (default is 0)
inp_node = fk_node.i()

inp_node.outputs = fk_node.outputs
fk_node.outputs.clear()
graph.cleanup()
model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
onnx.save(model, "removed.onnx")