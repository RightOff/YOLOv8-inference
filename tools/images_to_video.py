# -*- coding: utf-8 -*-
import cv2
import os
from natsort import natsorted

# 设置图片目录和输出视频文件名
image_folder = '/home/clh/data/fire_labelme/images'
video_name = 'output_video.mp4'

# 获取所有图片文件名并自然排序
images = natsorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

# 读取第一张图片以获取宽度和高度
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# 定义视频编写器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(video_name, fourcc, 60, (width, height))

# 将每张图片写入视频
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    videowriter.write(img)
    print(f'写入图片: {image}')

# 释放视频编写器
videowriter.release()
print('视频已生成: {}'.format(video_name))
