import numpy as np
import onnx
from onnx import TensorProto
# 这两个函数用于onnx和numpy之间的转换
from onnx.numpy_helper import from_array, to_array


np_tensor = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
print(type(np_tensor))

#
onnx_tensor = from_array(np_tensor)
print(type(onnx_tensor))

serialized_tensor = onnx_tensor.SerializeToString()
print(type(serialized_tensor))

with open("saved_tensor.pb", "wb") as f:
    f.write(serialized_tensor)

with open("saved_tensor.pb", "rb") as f:
    serialized_tensor = f.read()

# 创建onnx tensor
onnx_tensor = TensorProto()
onnx_tensor.ParseFromString(serialized_tensor)
print(type(onnx_tensor))

np_tensor = to_array(onnx_tensor)
print(np_tensor)

from onnx import load_tensor_from_string
'''
下面这一行可以代替
onnx_tensor = TensorProto()
onnx_tensor.ParseFromString(serialized_tensor)
'''
proto = load_tensor_from_string(serialized_tensor)
print(type(proto))