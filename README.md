# YOLOv8-inference
## 环境要求 
+ Jetpack 4.6.1 -> Ubuntu 18.04
+ cuda 10.2
+ cudnn 8.2.1
+ TensorRT 8.2.1
+ OpenCV 4.1.1
+ Cmake 3.10.2

## 概况
1. 自定义训练/预训练模型 -> pt
2. pt -> onnx -> onnx -> engine
4. cmake、make 编译

## 推理
1. pt -> onnx
```shell
cd tools
python3 export-det.py --weights yolov8n.pt --sim
```
2. onnx -> engine
在 Jetson 环境中
```shell
/usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine
# fp16, int8亦同
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```
3. 编译、推理
```shell
# 编译
mkdir build && cd build
cmake ..
make -j4

# 推理
./yolov8
```
## 其他
+ tools/images_to_video.py : 图片转换为 mp4