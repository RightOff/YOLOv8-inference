#include "mainwindow.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <chrono>

// 构造函数：初始化主窗口
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)  // 调用父类构造函数
    , detector(nullptr)     // 初始化检测器为 nullptr
    , isVideo(false)        // 初始化视频标志为 false
{
    setupUI();  // 设置用户界面
    timer = new QTimer(this);  // 创建定时器
    connect(timer, &QTimer::timeout, this, &MainWindow::processFrame);  // 连接定时器超时信号到处理帧的槽
}

// 析构函数：释放资源
MainWindow::~MainWindow()
{
    if (detector) {
        delete detector;  // 删除检测器对象
    }
}

// 设置用户界面
void MainWindow::setupUI()
{
    QWidget* centralWidget = new QWidget(this);  // 创建中央小部件
    setCentralWidget(centralWidget);  // 设置中央小部件

    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);  // 创建垂直布局
    
    // 创建图像显示区域
    imageLabel = new QLabel;  // 创建标签用于显示图像
    imageLabel->setMinimumSize(640, 480);  // 设置最小尺寸
    imageLabel->setAlignment(Qt::AlignCenter);  // 设置对齐方式
    imageLabel->setStyleSheet("QLabel { background-color: black; }");  // 设置背景颜色为黑色
    mainLayout->addWidget(imageLabel);  // 将标签添加到主布局

    // 创建推理时间显示区域
    inferenceTimeLabel = new QLabel("推理耗时: 0 ms", this);
    mainLayout->addWidget(inferenceTimeLabel);

    // 创建模型文件显示区域
    modelFileLabel = new QLabel("当前模型文件: ", this);
    mainLayout->addWidget(modelFileLabel);

    // 创建媒体文件显示区域
    mediaFileLabel = new QLabel("当前媒体文件: ", this);
    mainLayout->addWidget(mediaFileLabel);

    // 创建按钮区域
    QHBoxLayout* buttonLayout = new QHBoxLayout;  // 创建水平布局
    
    // 创建按钮
    selectModelButton = new QPushButton("选择模型文件", this);
    openImageBtn = new QPushButton("打开图片", this);  // 打开图片按钮
    openVideoBtn = new QPushButton("打开视频", this);  // 打开视频按钮
    startBtn = new QPushButton("开始检测", this);  // 开始检测按钮
    stopBtn = new QPushButton("停止检测", this);  // 停止检测按钮
    
    
    
    // 将按钮添加到按钮布局
    buttonLayout->addWidget(selectModelButton);
    buttonLayout->addWidget(openImageBtn);
    buttonLayout->addWidget(openVideoBtn);
    buttonLayout->addWidget(startBtn);
    buttonLayout->addWidget(stopBtn);
    
    mainLayout->addLayout(buttonLayout);  // 将按钮布局添加到主布局

    // 连接信号和槽
    connect(selectModelButton, &QPushButton::clicked, this, &MainWindow::selectModelFile);
    connect(openImageBtn, &QPushButton::clicked, this, &MainWindow::openImage);
    connect(openVideoBtn, &QPushButton::clicked, this, &MainWindow::openVideo);
    connect(startBtn, &QPushButton::clicked, this, &MainWindow::startDetection);
    connect(stopBtn, &QPushButton::clicked, this, &MainWindow::stopDetection);

    openImageBtn->setEnabled(false);
    openVideoBtn->setEnabled(false);
    stopBtn->setEnabled(false);
    startBtn->setEnabled(false);
    

    // 设置窗口标题和大小
    setWindowTitle("YOLOv8目标检测");
    resize(800, 600);
}

// 初始化模型
// 该函数负责初始化YOLOv8模型，包括选择engine文件和创建检测器对象。
void MainWindow::initializeModel()
{
    // 如果检测器为空，则初始化模型
    if (!detector) {
        // 选择engine文件
        QString engineFile = QFileDialog::getOpenFileName(this,
            tr("选择Engine文件"), "", tr("Engine Files (*.engine)"));
        
        if (engineFile.isEmpty()) {
            QMessageBox::warning(this, "错误", "未选择engine文件！");
            return;
        }
        
        enginePath = engineFile.toStdString();
        try {
            detector = new YOLOv8(enginePath);
            detector->make_pipe(true);
            startBtn->setEnabled(true);
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "错误", 
                QString("模型初始化失败：%1").arg(e.what()));
        }
    }
}

// 打开图片
// 该函数负责打开图片文件并显示在图像标签中。
void MainWindow::openImage()
{
    // 打开图片文件
    fileName = QFileDialog::getOpenFileName(this,
        tr("打开图片"), "", tr("Images (*.png *.jpg *.jpeg)"));
        
    if (!fileName.isEmpty()) {
        currentFrame = cv::imread(fileName.toStdString());
        if (!currentFrame.empty()) {
            QImage img = matToQImage(currentFrame);
            imageLabel->setPixmap(QPixmap::fromImage(img).scaled(
                imageLabel->size(), Qt::KeepAspectRatio));
            isVideo = false;
            mediaFileLabel->setText("当前媒体文件: " + fileName);
            initializeModel();
            startBtn->setEnabled(true);
            stopBtn->setEnabled(true);
        }
    }
}

// 打开视频
// 该函数负责打开视频文件并显示在图像标签中。
void MainWindow::openVideo()
{
    // 打开视频文件
    fileName = QFileDialog::getOpenFileName(this,
        tr("打开视频"), "", tr("Video Files (*.mp4 *.avi *.mkv)"));
        
    if (!fileName.isEmpty()) {
        cap.open(fileName.toStdString());
        if (cap.isOpened()) {
            isVideo = true;
            mediaFileLabel->setText("当前媒体文件: " + fileName);
            initializeModel();
            if (cap.read(currentFrame)) {
                QImage img = matToQImage(currentFrame);
                imageLabel->setPixmap(QPixmap::fromImage(img).scaled(
                    imageLabel->size(), Qt::KeepAspectRatio));
                startBtn->setEnabled(true);
                stopBtn->setEnabled(true);
            }
        }
    }
}

// 开始检测
// 该函数负责开始目标检测，包括启用定时器和设置按钮状态。
void MainWindow::startDetection()
{
    // 如果检测器为空，则提示初始化模型
    if (!detector) {
        QMessageBox::warning(this, "警告", "请先初始化模型！");
        return;
    }
    
    startBtn->setEnabled(false);
    stopBtn->setEnabled(true);
    openImageBtn->setEnabled(false);
    openVideoBtn->setEnabled(false);
    

    if (isVideo) {
        timer->start(33); // 定时器触发信号，执行推理
    } else {
        processFrame();
    }
}

// 停止检测
// 该函数负责停止目标检测，包括停止定时器和设置按钮状态。
void MainWindow::stopDetection()
{
    if (timer->isActive()) {
        timer->stop();
    }
    
    startBtn->setEnabled(true);
    stopBtn->setEnabled(false);
    openImageBtn->setEnabled(true);
    openVideoBtn->setEnabled(true);
}

// 处理帧
// 该函数负责处理视频帧或图片帧，包括读取帧、检测目标和显示结果。
void MainWindow::processFrame()
{
    if (isVideo) {
        if (!cap.read(currentFrame)) {
            cap.open(fileName.toStdString());
            stopDetection();
            return;
        }
    }

    if (currentFrame.empty()) return;

    try {
        cv::Mat processedFrame = currentFrame.clone();
        std::vector<Object> objects;
        
        detector->copy_from_Mat(currentFrame);
        
        auto start_infer = std::chrono::high_resolution_clock::now();
        
        detector->infer();

        auto end_infer = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> infer_duration = end_infer - start_infer;
        
        detector->postprocess(objects);
        
        // 使用你的绘制函数
        detector->draw_objects(currentFrame, processedFrame, objects, CLASS_NAMES, COLORS);
        
        QImage img = matToQImage(processedFrame);
        imageLabel->setPixmap(QPixmap::fromImage(img).scaled(
            imageLabel->size(), Qt::KeepAspectRatio));
            
        // 更新推理时间显示
        inferenceTimeLabel->setText(QString("推理耗时: %1 ms").arg(infer_duration.count()));
            
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", 
            QString("处理过程中发生错误：%1").arg(e.what()));
        stopDetection();
    }
    if (!isVideo){
        stopDetection();
    }

}

// 将OpenCV的Mat转换为QImage
// 该函数负责将OpenCV的Mat转换为QImage，用于显示在图像标签中。
QImage MainWindow::matToQImage(const cv::Mat& mat)
{
    if (mat.empty()) return QImage();

    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage((uchar*)rgb.data, rgb.cols, rgb.rows, 
            rgb.step, QImage::Format_RGB888).copy();
    }
    return QImage();
}

// 选择模型文件
void MainWindow::selectModelFile() {
    QString filePath = QFileDialog::getOpenFileName(this, "选择模型文件", "", "模型文件 (*.engine);;所有文件 (*.*)");
    if (!filePath.isEmpty()) {
        enginePath = filePath.toStdString();
        modelFileLabel->setText("当前模型文件: " + filePath);
        try {
            detector = new YOLOv8(enginePath);
            detector->make_pipe(true);
            openImageBtn->setEnabled(true);
            openVideoBtn->setEnabled(true);
    
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "错误", 
                QString("模型初始化失败：%1").arg(e.what()));
        }
    }
}
