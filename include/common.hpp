//
// YOLOv8-TensorRT公共头文件
// 创建时间：3/16/23
//

#ifndef JETSON_DETECT_COMMON_HPP
#define JETSON_DETECT_COMMON_HPP
#include "NvInfer.h"        // TensorRT核心头文件
#include "filesystem.hpp"   // 文件系统操作
#include "opencv2/opencv.hpp"  // OpenCV图像处理

// CUDA错误检查宏，用于检查CUDA API调用是否成功
#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// TensorRT日志记录器类，用于输出TensorRT的日志信息
class Logger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;  // 可报告的最低日志级别

    // 构造函数，设置日志级别，默认为INFO
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO):
        reportableSeverity(severity)
    {
    }

    // 日志输出函数
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // 如果日志级别低于设定级别，则不输出
        if (severity > reportableSeverity) {
            return;
        }
        // 根据日志级别输出不同的前缀
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
};

// 计算TensorRT维度的总大小（元素个数）
inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

// 获取TensorRT数据类型对应的字节大小
inline int type_to_size(const nvinfer1::DataType& dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:  // float32
            return 4;
        case nvinfer1::DataType::kHALF:   // float16
            return 2;
        case nvinfer1::DataType::kINT32:  // int32
            return 4;
        case nvinfer1::DataType::kINT8:   // int8
            return 1;
        case nvinfer1::DataType::kBOOL:   // bool
            return 1;
        default:
            return 4;
    }
}

// 将值限制在指定范围内
inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}

// 目标检测相关的数据结构
namespace det {

// TensorRT绑定信息结构体
struct Binding {
    size_t         size  = 1;    // 数据大小（元素个数）
    size_t         dsize = 1;    // 数据类型大小（字节数）
    nvinfer1::Dims dims;         // 数据维度
    std::string    name;         // 绑定名称
};

// 检测目标结构体
struct Object {
    cv::Rect_<float> rect;       // 目标框坐标（x,y,width,height）
    int              label = 0;   // 类别标签
    float            prob  = 0.0; // 置信度分数
};

// 预处理参数结构体
struct PreParam {
    float ratio  = 1.0f;  // 缩放比例
    float dw     = 0.0f;  // 宽度方向的填充量
    float dh     = 0.0f;  // 高度方向的填充量
    float height = 0;     // 原始图像高度
    float width  = 0;     // 原始图像宽度
};

}  // namespace det
#endif  // JETSON_DETECT_COMMON_HPP
