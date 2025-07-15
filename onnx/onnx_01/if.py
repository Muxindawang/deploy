import numpy as np
import onnx
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info
from onnx.numpy_helper import to_array, from_array
from onnx.checker import check_model
import onnxruntime

value = np.array([0], dtype=np.float32)
zero = from_array(value, name='zero')

# Same as before, X is the input, Y is the output.
X = make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None])

rsum = make_node('ReduceSum', ['X'], ['rsum'])
cond = make_node('Greater', ['rsum', 'zero'], ['cond'])

then_out = make_tensor_value_info('then_out', onnx.TensorProto.FLOAT, None)
then_cst = from_array(np.array([1]).astype(np.float32))

then_const_node = make_node('Constant', inputs=[], outputs=['then_out'], value=then_cst, name='cst1')

then_body = make_graph([then_const_node], 'then_body', [], [then_out])

else_out = make_tensor_value_info('else_out', onnx.TensorProto.FLOAT, None)
else_cst = from_array(np.array(-1).astype(np.float32))

else_const_node = make_node(
    'Constant', inputs=[],
    outputs=['else_out'],
    value=else_cst, name='cst2')

else_body = make_graph(
    [else_const_node], 'else_body',
    [], [else_out])

# Finally the node If taking both graphs as attributes.
if_node = onnx.helper.make_node(
    'If', ['cond'], ['Y'],
    then_branch=then_body,
    else_branch=else_body)
# The final graph.
graph = make_graph([rsum, cond, if_node], 'if', [X], [Y], [zero])
onnx_model = make_model(graph)
check_model(onnx_model)

# Let's freeze the opset.
del onnx_model.opset_import[:]
opset = onnx_model.opset_import.add()
opset.domain = ''
opset.version = 15
onnx_model.ir_version = 8

# Save.
with open("onnx_if_sign.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
x = np.zeros((3, 2), dtype=np.float32)
res = sess.run(None, {'X': x})

print("result: ", res)