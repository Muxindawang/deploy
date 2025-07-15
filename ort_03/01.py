import os
import wget

# 设置HuggingFace镜像源，提升国内下载速度
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 创建模型缓存目录
cache_dir = os.path.join('.', 'cache_models')
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

# 下载SQuAD数据集dev集
predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
predict_file = os.path.join(cache_dir, "dev-v1.1.json")
if not os.path.exists(predict_file):
    print("Start downloading predict file.")
    wget.download(predict_file_url, predict_file)
    print("Predict file downloaded.")

# 设定模型和数据处理参数
model_name_or_path = 'bert-base-cased'
max_seq_length = 128
doc_stride = 128
max_query_length = 64

enable_overwrite = True  # 是否每次都重新导出ONNX模型

total_samples = 100  # 用于测试的样本数
# 以下代码参考自HuggingFace transformers官方SQuAD脚本
# https://github.com/huggingface/transformers/blob/master/examples/run_squad.py

from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)

# 加载预训练模型和分词器
config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
config = config_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
model = model_class.from_pretrained(model_name_or_path,
                                    from_tf=False,
                                    config=config,
                                    cache_dir=cache_dir)
# 加载SQuAD数据集样本
from transformers.data.processors.squad import SquadV1Processor

processor = SquadV1Processor()
examples = processor.get_dev_examples(None, filename=predict_file)

from transformers import squad_convert_examples_to_features
# 将样本转为BERT输入特征和PyTorch数据集
features, dataset = squad_convert_examples_to_features(
            examples=examples[:total_samples], # 只处理部分样本
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            return_dataset='pt'
        )

# 创建ONNX模型输出目录
output_dir = os.path.join(".", "onnx_models")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
export_model_path = os.path.join(output_dir, 'bert-base-cased-squad.onnx')

import torch
device = torch.device("cpu")  # 只用CPU推理

# 取第一个样本，准备导出ONNX
data = dataset[0]
inputs = {
    'input_ids':      data[0].to(device).reshape(1, max_seq_length),
    'attention_mask': data[1].to(device).reshape(1, max_seq_length),
    'token_type_ids': data[2].to(device).reshape(1, max_seq_length)
}

# 设置模型为推理模式
model.eval()
model.to(device)

# 导出ONNX模型（如已存在且enable_overwrite为False则跳过）
if enable_overwrite or not os.path.exists(export_model_path):
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,                                            # 被导出的模型
                          args=tuple(inputs.values()),                      # 输入（按顺序）
                          f=export_model_path,                              # 导出文件路径
                          opset_version=14,                                 # ONNX算子集版本
                          do_constant_folding=True,                         # 是否常量折叠优化
                          input_names=['input_ids',                         # 输入名
                                       'input_mask',
                                       'segment_ids'],
                          output_names=['start', 'end'],                    # 输出名
                          dynamic_axes={'input_ids': symbolic_names,        # 动态维度
                                        'input_mask' : symbolic_names,
                                        'segment_ids' : symbolic_names,
                                        'start' : symbolic_names,
                                        'end' : symbolic_names})
        print("Model exported at ", export_model_path)


import time

# PyTorch推理性能测试
latency = []
with torch.no_grad():
    for i in range(total_samples):
        data = dataset[i]
        inputs = {
            'input_ids':      data[0].to(device).reshape(1, max_seq_length),
            'attention_mask': data[1].to(device).reshape(1, max_seq_length),
            'token_type_ids': data[2].to(device).reshape(1, max_seq_length)
        }
        start = time.time()
        outputs = model(**inputs)
        latency.append(time.time() - start)
print("PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))

import onnxruntime
import numpy

sess_options = onnxruntime.SessionOptions()

# 可选：保存优化后的ONNX模型，便于用Netron可视化调试
sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_cpu.onnx")

# 指定推理设备为CPU
session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])

# ONNXRuntime推理性能测试
latency = []
for i in range(total_samples):
    data = dataset[i]
    ort_inputs = {
        'input_ids':  data[0].cpu().reshape(1, max_seq_length).numpy(),
        'input_mask': data[1].cpu().reshape(1, max_seq_length).numpy(),
        'segment_ids': data[2].cpu().reshape(1, max_seq_length).numpy()
    }
    start = time.time()
    ort_outputs = session.run(None, ort_inputs)
    latency.append(time.time() - start)
print("OnnxRuntime cpu Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))