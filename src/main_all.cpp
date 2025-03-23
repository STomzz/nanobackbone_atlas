#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "acl/acl.h"
#include "label.h"
#include <chrono>
#include <vector>
#include <fstream>
#include <math.h>
#define INFO_LOG(fmt, ...)                               \
    fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define EXEMPLAR_SIZE 127
#define INSTANCE_SIZE 255
#define CONTEXT_AMOUT 0.5
#define OUTPUT_SIZE 15
#define STRIDE 16
#define INPUT0 0
#define INPUT1 1
#define times 1

// using namespace cv;
// using namespace std;

typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

class nanoTracker
{
public:
    nanoTracker(int32_t device, const char *ModelPath,
                std::vector<int32_t> inputWidths, std::vector<int32_t> inputHeights);
    ~nanoTracker();
    Result InitResource();
    Result ProcessInput(const std::vector<std::string> &testImgPaths);
    Result Input_preprocess(const cv::Mat &resizedImage, int i);
    Result Inference();
    Result GetResult();
    Result save_bin(const float *outputData, const size_t length, const char *filename);
    cv::Mat get_subWindow(const cv::Mat &im, const cv::Point2f &pos, int model_sz, int original_sz, const cv::Scalar &avg_chans);

    Result tracker_init(cv::Mat &img, const cv::Rect &bbox);
    Result tracker_track(cv::Mat frame);
    Result output_convert_score(const float *);
    Result generate_points(int stride, int size);
    Result output_convert_bbox(const float *, std::vector<cv::Point2f>);
    // 其他成员函数保持相似
private:
    void ReleaseResource();
    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    uint32_t modelId_;
    const char *modelPath_;
    std::vector<int32_t> modelWidths_;
    std::vector<int32_t> modelHeights_;
    aclmdlDesc *modelDesc_;
    aclmdlDataset *inputDataset_;
    aclmdlDataset *outputDataset_;

    std::vector<void *> inputBuffers_;
    std::vector<size_t> inputBufferSizes_;
    std::vector<void *> outputBuffers_;
    std::vector<size_t> outputBufferSizes_;

    std::vector<cv::Mat> srcImages_;
    aclrtRunMode runMode_;

    cv::Point2f center_pos_;
    cv::Size2f size_;
    cv::Scalar channel_average_;
    cv::Mat template_crop_;
    std::vector<std::vector<float>> score_;
    std::vector<cv::Point2f> points_;
};

nanoTracker::nanoTracker(int32_t device, const char *ModelPath,
                         std::vector<int32_t> inputWidths, std::vector<int32_t> inputHeights)
    : deviceId_(device), context_(nullptr), stream_(nullptr), modelId_(0),
      modelPath_(ModelPath), modelWidths_(inputWidths), modelHeights_(inputHeights),
      modelDesc_(nullptr), inputDataset_(nullptr), outputDataset_(nullptr)
{
    generate_points(STRIDE, OUTPUT_SIZE);
}

nanoTracker::~nanoTracker()
{
    ReleaseResource();
}

Result nanoTracker::InitResource()
{
    INFO_LOG("InitResource");
    // ... [保持原有初始化逻辑不变] ...
    const char *aclConfigPath = "";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclInit failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtSetDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtCreateContext failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtCreateStream failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = aclmdlLoadFromFile(modelPath_, &modelId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclmdlLoadFromFile failed, errorCode is %d", ret);
        return FAILED;
    }

    // create description of model
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclmdlGetDesc failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED)
    {
        ERROR_LOG("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // 创建输入数据集
    inputDataset_ = aclmdlCreateDataset();
    size_t numInputs = aclmdlGetNumInputs(modelDesc_);
    for (size_t i = 0; i < numInputs; ++i)
    {
        size_t bufferSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
        void *buffer;
        aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        inputBuffers_.push_back(buffer);
        inputBufferSizes_.push_back(bufferSize);

        aclDataBuffer *dataBuffer = aclCreateDataBuffer(buffer, bufferSize);
        aclmdlAddDatasetBuffer(inputDataset_, dataBuffer);
    }

    // 创建输出数据集
    outputDataset_ = aclmdlCreateDataset();
    size_t numOutputs = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < numOutputs; ++i)
    {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        void *buffer;
        aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        outputBuffers_.push_back(buffer);
        outputBufferSizes_.push_back(bufferSize);

        aclDataBuffer *dataBuffer = aclCreateDataBuffer(buffer, bufferSize);
        aclmdlAddDatasetBuffer(outputDataset_, dataBuffer);
    }
    return SUCCESS;
}
cv::Mat nanoTracker::get_subWindow(const cv::Mat &im, const cv::Point2f &pos, int model_sz, int original_sz, const cv::Scalar &avg_chans)
{
    INFO_LOG("get_subWindow");
    cv::Point2f center_position = pos;
    if (im.empty())
        return cv::Mat();
    int sz = original_sz;
    cv::Size im_sz = im.size();
    float c = (original_sz + 1) / 2.0f;

    float context_xmin = floor(center_position.x - c + 0.5);
    float context_xmax = context_xmin + sz - 1;
    float context_ymin = floor(center_position.y - c + 0.5);
    float context_ymax = context_ymin + sz - 1;

    int left_pad = static_cast<int>(std::max(0.0f, -context_xmin));
    int top_pad = static_cast<int>(std::max(0.0f, -context_xmax));
    int right_pad = static_cast<int>(std::max(0.0f, context_xmax - im_sz.width + 1));
    int bottom_pad = static_cast<int>(std::max(0.0f, context_ymax - im_sz.height + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat im_patch;
    if (top_pad > 0 || bottom_pad > 0 || left_pad > 0 || right_pad > 0)
    {
        cv::Mat te_im(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, im.type(), avg_chans);

        cv::Rect roi_rect(left_pad, top_pad, im.cols, im.rows);
        im.copyTo(te_im(roi_rect));

        int xmin = static_cast<int>(context_xmin);
        int ymin = static_cast<int>(context_ymin);
        int xmax = static_cast<int>(context_xmax) + 1;
        int ymax = static_cast<int>(context_ymax) + 1;

        xmin = std::max(xmin, 0);
        ymin = std::max(ymin, 0);
        xmax = std::min(xmax, te_im.cols);
        ymax = std::min(ymax, te_im.rows);

        im_patch = te_im(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    }
    else
    {
        int xmin = static_cast<int>(context_xmin);
        int ymin = static_cast<int>(context_ymin);
        int xmax = static_cast<int>(context_xmax) + 1;
        int ymax = static_cast<int>(context_ymax) + 1;

        xmin = std::max(xmin, 0);
        ymin = std::max(ymin, 0);
        xmax = std::min(xmax, im.cols);
        ymax = std::min(ymax, im.rows);

        im_patch = im(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    }

    if (model_sz != original_sz)
    {
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz), 0, 0, cv::INTER_LINEAR);
    }
    return im_patch;
}

Result nanoTracker::tracker_init(cv::Mat &img, const cv::Rect &bbox)
{
    INFO_LOG("tracker_init");
    center_pos_.x = bbox.x + (bbox.width - 1) / 2.0f;
    center_pos_.y = bbox.y + (bbox.height - 1) / 2.0f;
    size_ = cv::Size2f(bbox.width, bbox.height);

    float context = CONTEXT_AMOUT * (size_.width + size_.height);
    float w_z = size_.width + context;
    float h_z = size_.height + context;
    int s_z = static_cast<int>(std::round(std::sqrt(w_z * h_z)));

    channel_average_ = cv::mean(img);
    cv::Mat subWindow = get_subWindow(img, center_pos_, EXEMPLAR_SIZE, s_z, channel_average_);
    template_crop_ = nanoTracker::Input_preprocess(subWindow, INPUT0);
    return SUCCESS;
}

Result nanoTracker::Input_preprocess(const cv::Mat &resizedImage, int i)
{
    // cv::Mat resizedImage;
    // resize(srcImage, resizedImage, cv::Size(model_Width, model_height));
    INFO_LOG("Input_preprocess");
    int32_t channels = resizedImage.channels();
    int32_t resizeHeight = resizedImage.rows;
    int32_t resizeWidth = resizedImage.cols;
    assert(resizeHeight == resizeWidth);
    // 转换图像数据到NCHW格式
    float *imageBytes = (float *)malloc(channels * resizeHeight * resizeWidth * sizeof(float));
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < resizeHeight; ++h)
        {
            for (int w = 0; w < resizeWidth; ++w)
            {
                int idx = (c * resizeHeight + h) * resizeWidth + w;
                imageBytes[idx] = static_cast<float>(resizedImage.ptr<uchar>(h, w)[c]);
            }
        }
    }

    // 复制数据到设备
    aclrtMemcpyKind kind = runMode_ == ACL_DEVICE ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_HOST_TO_HOST;
    aclrtMemcpy(inputBuffers_[i], inputBufferSizes_[i], imageBytes, inputBufferSizes_[i], kind);
    free(imageBytes);

    return SUCCESS;
}

Result nanoTracker::tracker_track(cv::Mat frame)
{
    INFO_LOG("tracker_track");
    float context = CONTEXT_AMOUT * (size_.width + size_.height);
    float w_z = size_.width + context;
    float h_z = size_.height + context;
    int s_z = static_cast<int>(std::round(std::sqrt(w_z * h_z)));

    float scale_z = EXEMPLAR_SIZE / s_z;
    int s_x = static_cast<int>(std::round(s_z * (INSTANCE_SIZE / static_cast<float>(EXEMPLAR_SIZE))));
    cv::Mat x_Window = get_subWindow(frame, center_pos_, INSTANCE_SIZE, s_x, channel_average_);
    if (nanoTracker::Input_preprocess(x_Window, INPUT1) != SUCCESS)
    {
        ERROR_LOG("Input_preprocess failed");
        return FAILED;
    }
    if (nanoTracker::Inference() != SUCCESS)
    {
        ERROR_LOG("Inference failed");
        return FAILED;
    }

    return SUCCESS;
}

Result nanoTracker::output_convert_score(const float *output)
{
    INFO_LOG("output_convert_score");
    // softmax  1x2x15x15 => 2x225
    score_ = std::vector<std::vector<float>>(2, std::vector<float>(225));

    for (int scorenum = 0; scorenum < 225; scorenum++)
    {
        float maxValue = std::max(*(output + scorenum), *(output + 225 + scorenum));
        score_[0][scorenum] = std::exp(*(output + scorenum) - maxValue);
        score_[1][scorenum] = std::exp(*(output + 225 + scorenum) - maxValue);
        // score_[0][scorenum] /= (score_[0][scorenum] + score_[1][scorenum]);
        score_[1][scorenum] /= (score_[0][scorenum] + score_[1][scorenum]);
    }
    return SUCCESS;
}

Result nanoTracker::generate_points(int stride, int size)
{
    int ori = -(size / 2) * stride;

    for (int dy = 0; dy < size; ++dy)
    {
        for (int dx = 0; dx < size; ++dx)
        {
            float x = ori + dx * stride;
            float y = ori + dy * stride;
            points_.emplace_back(x, y);
        }
    }
    return SUCCESS;
}

Result nanoTracker::output_convert_bbox(const float *output, std::vector<cv::Point2f> points)
{
    INFO_LOG("output_convert_bbox");
    return SUCCESS;
}

Result nanoTracker::ProcessInput(const std::vector<std::string> &imgPaths)
{
    INFO_LOG("ProcessInput");
    // 处理两个输入
    for (size_t i = 0; i < imgPaths.size(); ++i)
    {
        cv::Mat srcImage = cv::imread(imgPaths[i]);
        cv::Mat resizedImage;
        resize(srcImage, resizedImage, cv::Size(modelWidths_[i], modelHeights_[i]));

        int32_t channels = resizedImage.channels();
        int32_t resizeHeight = resizedImage.rows;
        int32_t resizeWidth = resizedImage.cols;

        // 转换图像数据到NCHW格式
        float *imageBytes = (float *)malloc(channels * resizeHeight * resizeWidth * sizeof(float));
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < resizeHeight; ++h)
            {
                for (int w = 0; w < resizeWidth; ++w)
                {
                    int idx = (c * resizeHeight + h) * resizeWidth + w;
                    imageBytes[idx] = static_cast<float>(resizedImage.ptr<uchar>(h, w)[c]);
                }
            }
        }

        // 复制数据到设备
        aclrtMemcpyKind kind = runMode_ == ACL_DEVICE ? ACL_MEMCPY_HOST_TO_DEVICE : ACL_MEMCPY_HOST_TO_HOST;
        aclrtMemcpy(inputBuffers_[i], inputBufferSizes_[i], imageBytes, inputBufferSizes_[i], kind);
        free(imageBytes);
    }
    return SUCCESS;
}

Result nanoTracker::Inference()
{
    INFO_LOG("Inference");
    aclError ret = aclmdlExecute(modelId_, inputDataset_, outputDataset_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("Execute model failed, errorCode=%d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result nanoTracker::GetResult()
{
    INFO_LOG("GetResult");
    const std::vector<size_t> output_shapes = {1 * 2 * 15 * 15, 1 * 4 * 15 * 15};
    // const std::vector<const char *> filenames = {
    //     "/workspace/STomzz/nanobackbone_atlas/out/output1.bin",
    //     "/workspace/STomzz/nanobackbone_atlas/out/output2.bin"};
    for (size_t i = 0; i < outputBuffers_.size(); ++i)
    {
        void *outHostData = nullptr;
        aclrtMallocHost(&outHostData, outputBufferSizes_[i]);
        aclrtMemcpyKind kind = runMode_ == ACL_DEVICE ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_HOST;
        aclrtMemcpy(outHostData, outputBufferSizes_[i], outputBuffers_[i], outputBufferSizes_[i], kind);

        // 处理输出数据（示例）输出1 1*2*15*15 输出2 1*4*15*15
        float *outputData = reinterpret_cast<float *>(outHostData);
        // save to .bin
        try
        {
            // save_bin(outputData, output_shapes[i], filenames[i]);
            if (i == 0)
                output_convert_score(outputData);
            else
                output_convert_bbox(outputData);
        }
        catch (const std::exception &e)
        {
            aclrtFreeHost(outHostData);
            throw;
        }

        aclrtFreeHost(outHostData);
    }
    return SUCCESS;
}
Result nanoTracker::save_bin(const float *outputData, const size_t length, const char *filename)
{
    INFO_LOG("save_bin");
    std::ofstream out_file(filename, std::ios::binary);
    if (!out_file)
    {
        throw std::runtime_error("open file failed!");
    }
    out_file.write(reinterpret_cast<const char *>(outputData), length * sizeof(float));
    out_file.close();

    if (!out_file.good())
    {
        out_file.close();
        throw std::runtime_error("write file failed!");
    }

    return SUCCESS;
}

void nanoTracker::ReleaseResource()
{
    INFO_LOG("ReleaseResource");
    // 释放输入输出缓冲
    for (auto &buf : inputBuffers_)
    {
        aclrtFree(buf);
        buf = nullptr;
    }
    for (auto &buf : outputBuffers_)
    {
        aclrtFree(buf);
        buf = nullptr;
    }
    // ... [其余资源释放保持原逻辑] ...
    aclError ret;
    // release resource includes acl resource, data set and unload model
    (void)aclmdlDestroyDataset(inputDataset_);
    inputDataset_ = nullptr;

    (void)aclmdlDestroyDataset(outputDataset_);
    outputDataset_ = nullptr;

    ret = aclmdlDestroyDesc(modelDesc_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("destroy description failed, errorCode is %d", ret);
    }

    ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("unload model failed, errorCode is %d", ret);
    }

    if (stream_ != nullptr)
    {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("aclrtDestroyStream failed, errorCode is %d", ret);
        }
        stream_ = nullptr;
    }

    if (context_ != nullptr)
    {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("aclrtDestroyContext failed, errorCode is %d", ret);
        }
        context_ = nullptr;
    }

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtResetDevice failed, errorCode is %d", ret);
    }

    ret = aclFinalize();
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclFinalize failed, errorCode is %d", ret);
    }
}

int main()
{
    // const char* modelPath = "../models/nanotrack_all_2.om";
    // const char *modelPath = "../models/nanotrack_deploy_model.om";
    const char *modelPath = "../models/nanotrack_deploy_model_nchw.om";

    // std::vector<std::string> imagePaths = {"../data/000.png", "../data/001.png"};
    std::vector<int> inputWidths = {127, 255}; // 示例尺寸
    std::vector<int> inputHeights = {127, 255};

    nanoTracker tracker(0, modelPath, inputWidths, inputHeights);
    if (tracker.InitResource() != SUCCESS)
    {
        ERROR_LOG("Init failed");
        return FAILED;
    }

    /*功能:读取视频文件*/
    cv::VideoCapture capture("../video/original/input.mkv");
    cv::Mat frame;
    if (!capture.isOpened())
    {
        std::cerr << "Error: Failed to open video file!" << std::endl;
        return FAILED;
    }
    /*功能：选取目标图像*/
    capture >> frame;
    if (tracker.tracker_init(frame, cv::Rect(711, 680, 156, 127)) != SUCCESS)
    {
        ERROR_LOG("Init failed");
        return FAILED;
    }

    while (capture.read(frame))
    {
        std::cout << "=======read frame======" << std::endl;
        tracker.tracker_track(frame);
        tracker.GetResult();
    }
    capture.release();

    /*功能：将模板目标 和 候选区域 进行前处理*/
    // if (tracker.ProcessInput(imagePaths) != SUCCESS)
    // {
    //     ERROR_LOG("Process input failed");
    //     return FAILED;
    // }

    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < times; ++i)
    // {
    //     tracker.Inference();
    // }
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    //     std::chrono::high_resolution_clock::now() - start);
    // std::cout << "FPS: " << times * 1e6 / duration.count() << std::endl;

    // tracker.GetResult();
    return SUCCESS;
}