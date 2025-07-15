import onnx
from onnx import TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model

# inputs

# name, type, shape
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# outputs
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

# nodes
# 问题：op_type是不是可查询的，onnx已经实现的操作？可以自定义吗？
# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

# from node to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.
graph = make_graph([node1, node2],
                   'lr',
                   [X, A, B],
                   [Y])

# onnx graph
# there is no metadata in this case.
onnx_model = make_model(graph)

check_model(onnx_model)

# print(onnx_model)

# The serialization
with open('linear_regression.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

from onnx import load

with open('linear_regression.onnx', 'rb') as f:
    onnx_model = load(f)