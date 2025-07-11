import onnx
from onnx import helper
from onnx import TensorProto


a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])

mul = helper.make_node('Mul', ['a', 'x'], ['c'])
add = helper.make_node('Add', ['c', 'b'], ['output'])

graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])

model = helper.make_model(graph)
onnx.checker.check_model(model)
print(model)
onnx.save(model, 'linear_func.onnx')

import onnxruntime
import numpy as np
import onnx
model = onnx.load('linear_func.onnx')
print(model)
