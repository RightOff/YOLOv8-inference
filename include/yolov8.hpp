//
// YOLOv8目标检测模型的TensorRT实现
// 创建时间：3/16/23
//
#ifndef JETSON_DETECT_YOLOV8_HPP
#define JETSON_DETECT_YOLOV8_HPP
#include "NvInferPlugin.h"  // TensorRT插件头文件
#include "common.hpp"       // 通用工具函数和结构体定义
#include <fstream>
using namespace det;

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

// 构造函数实现：加载TensorRT引擎并初始化相关参数
YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    // 读取序列化的TensorRT引擎文件
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    // 将文件指针移动到文件末尾，用于获取文件大小
    file.seekg(0, std::ios::end);
    // 获取文件大小（字节数）
    auto size = file.tellg();
    // 将文件指针重新移动到文件开头
    file.seekg(0, std::ios::beg);
    // 分配足够的内存来存储整个模型文件
    char* trtModelStream = new char[size];
    // 确保内存分配成功
    assert(trtModelStream);
    // 读取整个文件内容到内存中
    file.read(trtModelStream, size);
    // 关闭文件
    file.close();

    // 初始化TensorRT插件和运行时
    // 注册所有内置的TensorRT插件到插件注册表中
    initLibNvInferPlugins(&this->gLogger, "");
    // 创建TensorRT运行时环境
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 反序列化引擎
    // 将内存中的序列化引擎数据反序列化为CUDA引擎
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    
    // 创建推理所需的资源
    // 创建执行上下文，用于保存模型的中间状态和临时数据
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);
    // 创建CUDA流，用于异步执行GPU操作
    cudaStreamCreate(&this->stream);
    
    // 获取模型的输入输出信息
    // 获取模型的总绑定数（输入+输出）
    this->num_bindings = this->engine->getNbBindings();

    // 遍历所有绑定，分别处理输入和输出
    for (int i = 0; i < this->num_bindings; ++i) {
        // 存储每个绑定的详细信息
        Binding            binding;
        nvinfer1::Dims     dims;
        // 获取绑定的数据类型（如FLOAT32, INT8等）
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        // 获取绑定的名称，用于区分不同的输入输出
        std::string        name  = this->engine->getBindingName(i);
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            // 处理输入绑定
            this->num_inputs += 1;
            dims         = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // 设置最大优化形状
            this->context->setBindingDimensions(i, dims);
        }
        else {
            // 处理输出绑定
            dims         = this->context->getBindingDimensions(i);  
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

// 析构函数：释放所有资源
YOLOv8::~YOLOv8()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    // 释放GPU内存
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    // 释放CPU内存
    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

// 初始化推理管道
void YOLOv8::make_pipe(bool warmup)
{
    // 为输入分配GPU内存
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMalloc(&d_ptr, bindings.size * bindings.dsize));
        this->device_ptrs.push_back(d_ptr);
    }

    // 为输出分配GPU和CPU内存
    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMalloc(&d_ptr, size));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));  //在主机上分配可页锁定（pinned）内存。
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // 模型预热
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                cudaStreamSynchronize(this->stream); 
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

// letterbox预处理：保持宽高比的图像缩放
void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    // 计算缩放比例
    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    // 缩放图像
    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    // 计算填充值
    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));    //？？为什么要偏移？
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    // 填充边界
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    // 转换为模型输入格式：NCHW，并归一化
    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    // 分离图像通道
    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    // 复制通道数据到输出
    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);

    // 归一化并复制数据
    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    // 保存预处理参数，用于后处理
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
}

// 从OpenCV Mat拷贝数据到模型输入
void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    int      width      = in_binding.dims.d[3];
    int      height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    // 设置输入维度
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});

    // 拷贝数据到GPU
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

// 从OpenCV Mat拷贝数据到模型输入（指定大小）
void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

// 执行模型推理
void YOLOv8::infer()
{
    // 执行推理
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    // 拷贝输出数据到CPU
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

// 后处理：将模型输出转换为检测框
void YOLOv8::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    int*  num_dets = static_cast<int*>(this->host_ptrs[0]);    // 检测到的目标数量
    auto* boxes    = static_cast<float*>(this->host_ptrs[1]);   // 边界框坐标
    auto* scores   = static_cast<float*>(this->host_ptrs[2]);   // 置信度分数
    int*  labels   = static_cast<int*>(this->host_ptrs[3]);     // 类别标签
    
    // 获取预处理参数
    auto& dw       = this->pparam.dw;
    auto& dh       = this->pparam.dh;
    auto& width    = this->pparam.width;
    auto& height   = this->pparam.height;
    auto& ratio    = this->pparam.ratio;

    // 处理每个检测框
    for (int i = 0; i < num_dets[0]; i++) {
        float* ptr = boxes + i * 4;

        // 还原检测框坐标到原图尺寸
        float x0 = *ptr++ - dw;
        float y0 = *ptr++ - dh;
        float x1 = *ptr++ - dw;
        float y1 = *ptr - dh;

        // 裁剪坐标到图像范围内
        x0 = clamp(x0 * ratio, 0.f, width);
        y0 = clamp(y0 * ratio, 0.f, height);
        x1 = clamp(x1 * ratio, 0.f, width);
        y1 = clamp(y1 * ratio, 0.f, height);

        // 保存检测结果
        Object obj;
        obj.rect.x      = x0;
        obj.rect.y      = y0;
        obj.rect.width  = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob        = *(scores + i);
        obj.label       = *(labels + i);
        objs.push_back(obj);
    }
}

// 在图像上绘制检测结果
void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    // 遍历每个检测框
    for (auto& obj : objs) {
        // 获取类别对应的颜色
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        // 绘制矩形框
        cv::rectangle(res, obj.rect, color, 2);

        // 准备标签文本
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        // 获取文本大小
        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        // 计算文本位置
        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }

        // 绘制文本背景
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        // 绘制文本
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}
#endif  // JETSON_DETECT_YOLOV8_HPP
