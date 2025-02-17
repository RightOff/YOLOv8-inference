#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <QMainWindow>
#include <QImage>
#include <QTimer>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include "yolov8.hpp"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    

private slots:
    void openImage();
    void openVideo();
    void startDetection();
    void stopDetection();
    void processFrame();
    void selectModelFile();
    void resizeEvent(QResizeEvent *event) override;
private:
    void setupUI();
    void initializeModel();
    QImage matToQImage(const cv::Mat& mat);

    QLabel* imageLabel;
    QLabel* inferenceTimeLabel;  // 用于显示推理时间
    QLabel* modelFileLabel;  // 用于显示模型文件名
    QLabel* mediaFileLabel;  // 用于显示媒体文件名
    QPushButton* openImageBtn;
    QPushButton* openVideoBtn;
    QPushButton* startBtn;
    QPushButton* stopBtn;
    QPushButton* selectModelButton;  // 用于选择模型文件

    cv::Mat currentFrame;
    cv::VideoCapture cap;
    QTimer* timer;
    YOLOv8* detector;
    bool isVideo;
    std::string enginePath;
    QString modelName;  // 模型文件名
    QString mediaName;  // 视频/图片名
    QString fileName; // 当前媒体文件名
};

#endif // MAINWINDOW_HPP
