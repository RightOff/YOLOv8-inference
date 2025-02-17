//
// YOLOv8目标检测模型的TensorRT实现
// 
//
#ifndef JETSON_DETECT_YOLOV8_HPP
#define JETSON_DETECT_YOLOV8_HPP
#include "NvInferPlugin.h"  // TensorRT插件头文件
#include "common.hpp"       // 通用工具函数和结构体定义
#include <fstream>
using namespace det;

// // COCO数据集的80个类别名称
// const std::vector<std::string> CLASS_NAMES = {
//     "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
//     "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
//     "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
//     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
//     "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
//     "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
//     "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
//     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
//     "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
//     "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
//     "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
//     "teddy bear",     "hair drier", "toothbrush"};
// 火焰烟雾数据集
const std::vector<std::string> CLASS_NAMES = {
    "fire", "smoke"};

// 用于目标检测可视化的颜色列表，每个类别对应一个RGB颜色
const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

// YOLOv8类：实现YOLOv8目标检测模型的推理功能
class YOLOv8 {
public:
    // 构造函数，加载TensorRT引擎文件
    explicit YOLOv8(const std::string& engine_file_path);
    // 析构函数，释放资源
    ~YOLOv8();

    // 初始化推理管道，可选是否进行预热
    void make_pipe(bool warmup = true);
    // 从OpenCV Mat格式图像拷贝数据到模型输入
    void copy_from_Mat(const cv::Mat& image);
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);
    // 对输入图像进行letterbox处理，保持宽高比
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    // 执行模型推理
    void infer();
    // 对模型输出进行后处理，得到检测结果
    void postprocess(std::vector<Object>& objs);
    // 在图像上绘制检测结果
    static void draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<Object>&                    objs,
                             const std::vector<std::string>&               CLASS_NAMES,
                             const std::vector<std::vector<unsigned int>>& COLORS);

    // TensorRT相关参数
    int                  num_bindings;  // 绑定数量（输入+输出）
    int                  num_inputs  = 0;  // 输入数量
    int                  num_outputs = 0;  // 输出数量
    std::vector<Binding> input_bindings;   // 输入绑定信息
    std::vector<Binding> output_bindings;  // 输出绑定信息
    std::vector<void*>   host_ptrs;        // CPU内存指针
    std::vector<void*>   device_ptrs;      // GPU内存指针

    PreParam pparam;  // 预处理参数

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;  // TensorRT引擎
    nvinfer1::IRuntime*          runtime = nullptr;  // TensorRT运行时
    nvinfer1::IExecutionContext* context = nullptr;  // TensorRT执行上下文
    cudaStream_t                 stream  = nullptr;  // CUDA流
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};  // TensorRT日志记录器
};

#endif  // JETSON_DETECT_YOLOV8_HPP
