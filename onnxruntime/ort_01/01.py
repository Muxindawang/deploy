import numpy as np
import onnxruntime
import onnx
from bokeh.io import output_file
from onnx import numpy_helper
import urllib.request
import json
import time
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


onnx_model_url = "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz"
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# download model and label
if not os.path.exists('resnet50v2.tar.gz'):
    urllib.request.urlretrieve(onnx_model_url, filename='resnet50v2.tar.gz')
if not os.path.exists('imagenet-simple-labels.json'):
    urllib.request.urlretrieve(imagenet_labels_url, filename='imagenet-simple-labels.json')

test_data_dir = 'resnet50v2/test_data_set'
test_data_num = 3

inputs = []
for i in range(test_data_num):
    input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
    # 创建或修改 ONNX 模型中的权重（常量)
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(tensor))

print('Load {} reference outputs successfully'.format(test_data_num))


ref_outputs = []
for i in range(test_data_num):
    output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(tensor))

print('Load {} reference outputs successfully.'.format(test_data_num))

session = onnxruntime.InferenceSession('resnet50v2/resnet50v2.onnx', None)
input_name = session.get_inputs()[0].name
print('Input name: ', input_name)

outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]
print('Predicted {} results.'.format(len(outputs)))
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o, 4)
print('ONNX Runtime outputs are similar to reference outputs!')

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)

def postprocess(result):
    return softmax(np.array(result)).tolist()


labels = load_labels('imagenet-simple-labels.json')
image = Image.open('dog.jpg')

print("Image size: ", image.size)
plt.axis('off')
display_image = plt.imshow(image)
image_data = np.array(image).transpose(2, 0, 1)
image_data = image_data[:3, :, :]
input_data = preprocess(image_data)

start = time.time()
raw_result = session.run([], {input_name: input_data})
end = time.time()
res = postprocess(raw_result)

inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(res)

print('========================================')
print('Final top prediction is: ' + labels[idx])
print('========================================')

print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')

sort_idx = np.flip(np.squeeze(np.argsort(res)))
print('============ Top 5 labels are: ============================')
print(labels[sort_idx[:5]])
print('===========================================================')

plt.axis('off')
display_image = plt.imshow(image)