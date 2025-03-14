#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "acl/acl.h"
#include "label.h"
#include <chrono>
#include <vector>
#define INFO_LOG(fmt, ...)                               \
    fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define times 10000

using namespace cv;
using namespace std;

typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

class nanoTracker
{
public:
    nanoTracker(int32_t device, const char *ModelPath,
                vector<int32_t> inputWidths, vector<int32_t> inputHeights);
    ~nanoTracker();
    Result InitResource();
    Result ProcessInput(const vector<string> &testImgPaths);
    Result Inference();
    Result GetResult();
    // 其他成员函数保持相似
private:
    void ReleaseResource();
    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    uint32_t modelId_;
    const char *modelPath_;
    vector<int32_t> modelWidths_;
    vector<int32_t> modelHeights_;
    aclmdlDesc *modelDesc_;
    aclmdlDataset *inputDataset_;
    aclmdlDataset *outputDataset_;

    vector<void *> inputBuffers_;
    vector<size_t> inputBufferSizes_;
    vector<void *> outputBuffers_;
    vector<size_t> outputBufferSizes_;

    vector<Mat> srcImages_;
    aclrtRunMode runMode_;
};

nanoTracker::nanoTracker(int32_t device, const char *ModelPath,
                         vector<int32_t> inputWidths, vector<int32_t> inputHeights)
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

Result nanoTracker::ProcessInput(const vector<string> &imgPaths)
{
    // 处理两个输入
    for (size_t i = 0; i < imgPaths.size(); ++i)
    {
        Mat srcImage = imread(imgPaths[i]);
        Mat resizedImage;
        resize(srcImage, resizedImage, Size(modelWidths_[i], modelHeights_[i]));

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
    for (size_t i = 0; i < outputBuffers_.size(); ++i)
    {
        void *outHostData = nullptr;
        aclrtMallocHost(&outHostData, outputBufferSizes_[i]);
        aclrtMemcpyKind kind = runMode_ == ACL_DEVICE ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_HOST_TO_HOST;
        aclrtMemcpy(outHostData, outputBufferSizes_[i], outputBuffers_[i], outputBufferSizes_[i], kind);

        // 处理输出数据（示例）
        float *outputData = reinterpret_cast<float *>(outHostData);
        // TODO: 根据实际需求处理输出

        aclrtFreeHost(outHostData);
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
    const char *modelPath = "../models/nanotrack_deploy_model.om";
    vector<string> imagePaths = {"../data/000.png", "../data/001.png"};
    vector<int> inputWidths = {127, 255}; // 示例尺寸
    vector<int> inputHeights = {127, 255};

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

    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < times; ++i)
    {
        tracker.Inference();
    }
    auto duration = chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now() - start);
    cout << "FPS: " << times * 1e6 / duration.count() << endl;

    tracker.GetResult();
    return SUCCESS;
}