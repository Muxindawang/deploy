from onnxruntime_extensions import onnx_op, PyOp

@onnx_op(op_type="Add", domain="Custom", inputs=[PyOp.dt_float, PyOp.dt_float], outputs=[PyOp.dt_float])
def custom_add(x, y):
    return x + y