import os
import requests

# 下载模型权重和测试图片，如果本地不存在的话
urls = ['https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth',
        'https://raw.githubusercontent.com/open-mmlab/mmediting/master/tests/data/face/000001.png']
names = ['srcnn.pth', 'face.png']
for url, name in zip(urls, names):
    if not os.path.exists(name):
        # 下载文件并保存到本地
        open(name, 'wb').write(requests.get(url).content)
