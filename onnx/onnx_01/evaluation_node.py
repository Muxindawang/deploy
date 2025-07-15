import numpy as np
from onnx import numpy_helper, TensorProto
from onnx.helper import make_node

from onnx.reference import ReferenceEvaluator

node = make_node('EyeLike', ['X'], ['Y'])

sess = ReferenceEvaluator(node)

x = np.random.randn(10, 10).astype(np.float32)
print(sess.run(None, {'X': x}))