from onnx import load

with open("linear_regression.onnx", "rb") as f:
    onnx_model = load(f)

# for field in ['doc_string', 'domain', 'functions',
#               'ir_version', 'metadata_props', 'model_version',
#               'opset_import', 'producer_name', 'producer_version',
#               'training_info']:
#     print(field, getattr(onnx_model, field))