// YOLOv8目标检测程序的主文件
// 创建时间：3/16/23

#include "opencv2/opencv.hpp"  // OpenCV库，用于图像处理
#include "yolov8.hpp"         // YOLOv8模型相关的头文件
#include <chrono>             // 用于时间测量

namespace fs = ghc::filesystem;  // 文件系统命名空间别名

// COCO数据集的80个类别名称
const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

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

int main(int argc, char** argv)
{
    // 检查命令行参数
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // 设置CUDA设备为0号设备
    cudaSetDevice(0);

    // 获取TensorRT引擎文件路径和输入文件路径
    const std::string engine_file_path{argv[1]};
    const fs::path    path{argv[2]};

    std::vector<std::string> imagePathList;  // 存储待处理图片路径
    bool                     isVideo{false};  // 标记输入是否为视频

    // 初始化YOLOv8模型
    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    // 处理输入路径
    if (fs::exists(path)) {
        std::string suffix = path.extension();
        // 判断是单张图片
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path);
        }
        // 判断是视频文件
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
                 || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    // 如果是目录，获取目录下所有jpg图片
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat             res, image;          // 结果图像和输入图像
    cv::Size            size = cv::Size{640, 640};  // 模型输入大小
    std::vector<Object> objs;                // 存储检测结果

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    // 处理视频
    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        // 逐帧处理视频
        while (cap.read(image)) {
            objs.clear();
            yolov8->copy_from_Mat(image, size);  // 预处理图像
            auto start = std::chrono::system_clock::now();
            yolov8->infer();  // 模型推理
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);  // 后处理获取检测结果
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);  // 绘制检测结果
            // 计算并打印推理时间
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            // 按q键退出
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    // 处理图片
    else {
        // 遍历处理每张图片
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8->copy_from_Mat(image, size);  // 预处理图像
            auto start = std::chrono::system_clock::now();
            yolov8->infer();  // 模型推理
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);  // 后处理获取检测结果
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);  // 绘制检测结果
            // 计算并打印推理时间
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    // 清理资源
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
