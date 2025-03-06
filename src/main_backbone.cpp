#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "acl/acl.h"
#include "label.h"
#include <chrono>
#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__);fflush(stdout)
#define ERROR_LOG(fmt, ...)fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define times 10000

using namespace cv;
using namespace std;

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

class nanoTracker{
    public:
        nanoTracker(int32_t device, const char* ModelPath,
        int32_t modelWidth, int32_t modelHeight);
        ~nanoTracker();
        Result InitResource();
        Result ProcessInput(const string testImgPath);
        Result Inference();
        Result GetResult();
        Result Conv(float*output1, float*output2);
    private:
        void ReleaseResource();
        int32_t deviceId_;
        aclrtContext context_;
        aclrtStream stream_;

        uint32_t modelId_;
        const char* modelPath_;
        int32_t modelWidth_;
        int32_t modelHeight_;
        aclmdlDesc *modelDesc_;
        aclmdlDataset *inputDataset_;
        aclmdlDataset *outputDataset_;
        void* inputBuffer_;
        void *outputBuffer_;
        size_t inputBufferSize_;
        float* imageBytes;
        String imagePath;
        Mat srcImage;
        aclrtRunMode runMode_;
};

nanoTracker::nanoTracker(int32_t device, const char* ModelPath,
        int32_t modelWidth, int32_t modelHeight):
    deviceId_(device), context_(nullptr), stream_(nullptr), modelId_(0),
    modelPath_(ModelPath), modelWidth_(modelWidth), modelHeight_(modelHeight),
    modelDesc_(nullptr), inputDataset_(nullptr), outputDataset_(nullptr){}

nanoTracker::~nanoTracker(){
    ReleaseResource();
}

Result nanoTracker::InitResource(){
    const char * aclConfigPath = "";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclInit failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtSetDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateContext failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtCreateStream failed, errorCode is %d", ret);
        return FAILED;
    }

     // load model from file
    ret = aclmdlLoadFromFile(modelPath_, &modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlLoadFromFile failed, errorCode is %d", ret);
        return FAILED;
    }

    // create description of model
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlGetDesc failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED) {
        ERROR_LOG("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

     // create data set of input
    inputDataset_ = aclmdlCreateDataset();
    size_t inputIndex = 0;
    inputBufferSize_ = aclmdlGetInputSizeByIndex(modelDesc_, inputIndex);
    aclrtMalloc(&inputBuffer_, inputBufferSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *inputData = aclCreateDataBuffer(inputBuffer_, inputBufferSize_);
    ret = aclmdlAddDatasetBuffer(inputDataset_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer failed, errorCode is %d", ret);
        return FAILED;
    }

    // create data set of output
    outputDataset_ = aclmdlCreateDataset();
    size_t outputIndex = 0;
    size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, outputIndex);
    aclrtMalloc(&outputBuffer_, modelOutputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer_, modelOutputSize);
    ret = aclmdlAddDatasetBuffer(outputDataset_, outputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlAddDatasetBuffer failed, errorCode is %d", ret);
        return FAILED;
    }
    
    return SUCCESS;
}

//model->input : 1x3x255x255 nchw  
Result nanoTracker::ProcessInput(const string ImgPath){
    imagePath = ImgPath;
    srcImage = imread(ImgPath);
    Mat resizedImage;
    resize(srcImage,resizedImage,Size(modelWidth_,modelHeight_));

    int32_t CHANNELS = resizedImage.channels();
    int32_t resizeHeight = resizedImage.rows;
    int32_t resizeWeight = resizedImage.cols;

    //nhwc -> nchw   bgr ? rgb
    imageBytes = (float*)malloc(CHANNELS * resizeHeight * resizeWeight * sizeof(float));
    memset(imageBytes, 0, CHANNELS * resizeHeight * resizeWeight * sizeof(float));
    
    for(int row = 0; row < resizeHeight; row++){
        for(int col = 0; col < resizeWeight; col++){
            for(int channel = 0; channel < CHANNELS; channel++){
                int new_idx = (channel*resizeHeight+row)*resizeWeight+col;
                // std::cout << "img" <<  static_cast<float>(img.ptr<uchar>(row, col)[channel]) << std::endl;
                imageBytes[new_idx] = static_cast<float>(resizedImage.ptr<uchar>(row, col)[channel]);
                // input_data[new_idx] = static_cast<float>(img.data[old_idx]);
            }
        }
    }


    return SUCCESS;
}

Result nanoTracker::Inference(){
    aclrtMemcpyKind kind;
    if (runMode_ == ACL_DEVICE)
    {
        kind = ACL_MEMCPY_DEVICE_TO_HOST;
    }else{
        kind = ACL_MEMCPY_HOST_TO_DEVICE;
    }
    aclError ret = aclrtMemcpy(inputBuffer_, inputBufferSize_, imageBytes, inputBufferSize_, kind);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = aclmdlExecute(modelId_, inputDataset_, outputDataset_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result nanoTracker::GetResult(){
    void *outHostData = nullptr;
    float *outData = nullptr;
    size_t outputIndex = 0;
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(outputDataset_, outputIndex);
    void* data = aclGetDataBufferAddr(dataBuffer);
    uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

    // copy device output data to host
    aclrtMemcpyKind kind;
    if (runMode_ == ACL_DEVICE)
    {
        kind = ACL_MEMCPY_DEVICE_TO_HOST;
    }else{
        kind = ACL_MEMCPY_HOST_TO_DEVICE;
    }
    aclrtMallocHost(&outHostData, len);
    aclError ret = aclrtMemcpy(outHostData, len, data, len, kind);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }
    outData = reinterpret_cast<float*>(outHostData);

 

    ret = aclrtFreeHost(outHostData);
    outHostData = nullptr;
    outData = nullptr;
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtFreeHost failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

void nanoTracker::ReleaseResource(){
    aclError ret;
    // release resource includes acl resource, data set and unload model
    aclrtFree(inputBuffer_);
    inputBuffer_ = nullptr;
    (void)aclmdlDestroyDataset(inputDataset_);
    inputDataset_ = nullptr;

    aclrtFree(outputBuffer_);
    outputBuffer_ = nullptr;
    (void)aclmdlDestroyDataset(outputDataset_);
    outputDataset_ = nullptr;

    ret = aclmdlDestroyDesc(modelDesc_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("destroy description failed, errorCode is %d", ret);
    }

    ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, errorCode is %d", ret);
    }

    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclrtDestroyStream failed, errorCode is %d", ret);
        }
        stream_ = nullptr;
    }

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclrtDestroyContext failed, errorCode is %d", ret);
        }
        context_ = nullptr;
    }

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtResetDevice failed, errorCode is %d", ret);
    }

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclFinalize failed, errorCode is %d", ret);
    }
}

//output1:1x96x16x16   output2 as kernel :1x96x8x8
Result nanoTracker::Conv(float*output1, float*output2){
    int out_H = 9;
    int out_W = 9;
    int out_C = 96;
    for(int out_h = 0; out_h < out_H; out_h++){
        for(int out_w = 0; out_w < out_W; out_w++){
            for(int out_c = 0;out_c < out_C;out_c++){

                


            }
        }

    }
}


int main(){

    const char * modelPath = "../models/nanotrack_backbone_om.om";
    const string imagePath = "../data/000.png";
    int32_t device = 0;
    int32_t modelWidth = 255;
    int32_t modelHeight = 255;


    
    //start
    nanoTracker nanobackbone(device, modelPath, modelWidth, modelHeight);
    Result ret = nanobackbone.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("InitResource  failed");
        return FAILED;
    }
    
    //开始计时
    auto start = std::chrono::high_resolution_clock::now();

    ret = nanobackbone.ProcessInput(imagePath);
    if (ret != SUCCESS) {
            ERROR_LOG("ProcessInput  failed");
            return FAILED;
    }


    for(int i = 0; i<times;i++){
        ret = nanobackbone.Inference();
    if (ret != SUCCESS) {
            ERROR_LOG("Inference  failed");
            return FAILED;
        }
    }
    
    //结束计时
    auto end = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<times<< "轮消耗时间:--"<< duration.count() <<std::endl;
    std::cout<<times<< "轮测量帧率:--"<< times * 1000000.0 /duration.count() <<std::endl;

    ret = nanobackbone.GetResult();
    if (ret != SUCCESS) {
            ERROR_LOG("GetResult  failed");
            return FAILED;
        }
    return SUCCESS;

}