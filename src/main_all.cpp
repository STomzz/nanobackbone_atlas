#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "acl/acl.h"
#include "label.h"
#include <chrono>
#include <vector>
#include <fstream>
#define INFO_LOG(fmt, ...)                               \
    fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
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
    Result Inference();
    Result GetResult();
    Result save_bin(const float *outputData, const size_t length, const char *filename);
    // cv::Mat get_subWindow(const cv::Mat &img, const cv::Point2f &position, int model_sz, const cv::Scalar &avg_chans);
    // Result tracker_init();
    // Result tracker_track();
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
};

nanoTracker::nanoTracker(int32_t device, const char *ModelPath,
                         std::vector<int32_t> inputWidths, std::vector<int32_t> inputHeights)
    : deviceId_(device), context_(nullptr), stream_(nullptr), modelId_(0),
      modelPath_(ModelPath), modelWidths_(inputWidths), modelHeights_(inputHeights),
      modelDesc_(nullptr), inputDataset_(nullptr), outputDataset_(nullptr) {}

nanoTracker::~nanoTracker()
{
    ReleaseResource();
}

Result nanoTracker::InitResource()
{
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
// cv::Mat nanoTracker::get_subWindow(const cv::Mat &im, const cv::Point2f &pos, int model_sz, int original_sz, const cv::Scalar &avg_chans)
// {
//     cv::Point2f center_pos = pos;
//     if (im.empty())
//         return cv::Mat();
//     int sz = original_sz;
//     cv::Size im_sz = im.size();
//     float c = (original_sz + 1) / 2.0f;

//     float context_xmin = floor(center_pos.x - c + 0.5);
//     float context_xmax = context_xmin + sz - 1;
//     float context_ymin = floor(center_pos.y - c + 0.5);
//     float context_ymax = context_ymin + sz - 1;

//     int left_pad = static_cast<int>(max(0.0f, -context_xmin));
//     int top_pad = static_cast<int>(max(0.0f, -context_xmax));
//     int right_pad = static_cast<int>(max(0.0f, context_xmax - im_sz.width + 1));
//     int bottom_pad = static_cast<int>(max(0.0f, context_ymax - im_sz.height + 1));

//     context_xmin += left_pad;
//     context_xmax += left_pad;
//     context_ymin += top_pad;
//     context_ymax += top_pad;

//     cv::Mat im_patch;
//     if (top_pad > 0 || bottom_pad > 0 || left_pad > 0 || right_pad > 0)
//     {
//         cv::Mat te_im(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, im.type(), avg_chans);

//         cv::Rect roi_rect(left_pad, top_pad, im.cols, im.rows);
//         im.copyTo(te_im(roi_rect));

//         int xmin = static_cast<int>(context_xmin);
//         int ymin = static_cast<int>(context_ymin);
//         int xmax = static_cast<int>(context_xmax) + 1;
//         int ymax = static_cast<int>(context_ymax) + 1;

//         xmin = std::max(xmin, 0);
//         ymin = std::max(ymin, 0);
//         xmax = std::min(xmax, te_im.cols);
//         ymax = std::min(ymax, te_im.rows);

//         im_patch = te_im(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
//     }
//     else
//     {
//         int xmin = static_cast<int>(context_xmin);
//         int ymin = static_cast<int>(context_ymin);
//         int xmax = static_cast<int>(context_xmax) + 1;
//         int ymax = static_cast<int>(context_ymax) + 1;

//         xmin = std::max(xmin, 0);
//         ymin = std::max(ymin, 0);
//         xmax = std::min(xmax, im.cols);
//         ymax = std::min(ymax, im.rows);

//         im_patch = im(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
//     }

//     if(model_sz != original_sz){
//         cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz), 0, 0, cv::INTER_LINEAR);
//     }
// }

Result nanoTracker::ProcessInput(const std::vector<std::string> &imgPaths)
{
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
    const std::vector<size_t> output_shapes = {1 * 2 * 15 * 15, 1 * 4 * 15 * 15};
    const std::vector<const char *> filenames = {
        "/workspace/STomzz/nanobackbone_atlas/out/output1.bin",
        "/workspace/STomzz/nanobackbone_atlas/out/output2.bin"};
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
            save_bin(outputData, output_shapes[i], filenames[i]);
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
    std::vector<std::string> imagePaths = {"../data/000.png", "../data/001.png"};
    std::vector<int> inputWidths = {127, 255}; // 示例尺寸
    std::vector<int> inputHeights = {127, 255};

    nanoTracker tracker(0, modelPath, inputWidths, inputHeights);
    if (tracker.InitResource() != SUCCESS)
    {
        ERROR_LOG("Init failed");
        return FAILED;
    }

    if (tracker.ProcessInput(imagePaths) != SUCCESS)
    {
        ERROR_LOG("Process input failed");
        return FAILED;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; ++i)
    {
        tracker.Inference();
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "FPS: " << times * 1e6 / duration.count() << std::endl;

    tracker.GetResult();
    return SUCCESS;
}