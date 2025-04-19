#include "ACNNModel_B.hpp"
#include <iostream>
// using namespace std;

ACNNModel_B::ACNNModel_B(const char *modelPath) : modelPath_(modelPath)
{
    INFO_LOG("START ACNNModel_B::ACNNModel_B ");
    aclError ret;
    // load model from file
    ret = aclmdlLoadFromFile(modelPath_, &modelId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclmdlLoadFromFile failed, errorCode is %d", ret);
    }
    // create description of model
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclmdlGetDesc failed, errorCode is %d", ret);
    }
    INFO_LOG("FINISH ACNNModel_B::ACNNModel_B ");
}
ACNNModel_B::~ACNNModel_B()
{
    aclError ret;
    // release resource includes acl resource, data set and unload model
    aclrtFree(inputBuffer_b);
    inputBuffer_b = nullptr;
    (void)aclmdlDestroyDataset(inputDataset_b);
    inputDataset_b = nullptr;

    aclrtFree(outputBuffer_b);
    outputBuffer_b = nullptr;
    (void)aclmdlDestroyDataset(outputDataset_b);
    outputDataset_b = nullptr;

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
}

Result ACNNModel_B::backbone_initDatasets()
{
    INFO_LOG("START backbone_initDatasets ");
    aclError ret;
    // create data set of input
    inputDataset_b = aclmdlCreateDataset();
    inputBufferSize_b = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    aclrtMalloc(&inputBuffer_b, inputBufferSize_b, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *inputData = aclCreateDataBuffer(inputBuffer_b, inputBufferSize_b);
    ret = aclmdlAddDatasetBuffer(inputDataset_b, inputData);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("backbone_initDatasets inputDataset_b failed, errorCode is %d", ret);
        return FAILED;
    }
    else
    {
        INFO_LOG("backbone_initDatasets inputDataset_b success");
    }

    // create data set of output
    outputDataset_b = aclmdlCreateDataset();
    modelOutputSize_b = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
    aclrtMalloc(&outputBuffer_b, modelOutputSize_b, ACL_MEM_MALLOC_HUGE_FIRST);
    aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer_b, modelOutputSize_b);
    ret = aclmdlAddDatasetBuffer(outputDataset_b, outputData);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("backbone_initDatasets outputDataset_b failed, errorCode is %d", ret);
        return FAILED;
    }
    else
    {
        INFO_LOG("backbone_initDatasets outputDataset_b success");
    }
    INFO_LOG("FINISH backbone_initDatasets ");
    return SUCCESS;
}

/*
1. 不进行标准化
2. HWC_2_CHW
3. CHW_2_NCHW
4. INT8_2_FLOAT32
5. 从多维数组(shape为[1,3,127,127])转成一维数组(字节bytes)
*/
Result ACNNModel_B::backbone_ProcessInput(cv::Mat &img)
{
    INFO_LOG("START Preprocess the input img ");

    // get properties of image
    int32_t channel = img.channels();
    int32_t Height = img.rows;
    int32_t Weight = img.cols;
    imageBytes = (float *)malloc(1 * channel * Height * Weight * sizeof(float));
    std::memset(imageBytes, 0, 1 * channel * Height * Weight * sizeof(float));

    // 图像转换为字节，从 HWC 到 NCHW
    for (int h = 0; h < Height; ++h)
    {
        for (int w = 0; w < Weight; ++w)
        {
            for (int c = 0; c < channel; ++c)
            {
                // 将像素值从 cv::Vec3b (即 uint8_t) 转换为 float
                imageBytes[c * Height * Weight + h * Weight + w] = static_cast<float>(img.at<cv::Vec3b>(h, w)[c]);
            }
        }
    }

    INFO_LOG("FINISH Preprocess the input img ");
    return SUCCESS;
}

Result ACNNModel_B::backbone_Inference()
{
    INFO_LOG("START ACNNModel_B::backbone_Inference");
    // copy host datainputs to device
    aclError ret = aclrtMemcpy(inputBuffer_b, inputBufferSize_b, this->imageBytes, inputBufferSize_b, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }
    // inference
    ret = aclmdlExecute(modelId_, inputDataset_b, outputDataset_b);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    INFO_LOG("FINISH ACNNModel_B::backbone_Inference");

    return SUCCESS;
}
Result ACNNModel_B::backbone_GetResults(std::vector<std::vector<float>> &output)
{
    INFO_LOG("START ACNNModel_B::backbone_GetResults");
    aclError ret;
    void *outHostData = nullptr;
    float *outData = nullptr;
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(outputDataset_b, 0);
    void *data = aclGetDataBufferAddr(dataBuffer);
    uint32_t output_length = aclGetDataBufferSizeV2(dataBuffer);

    // 查询模型输出数据的维度与数据类型
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(modelDesc_, 0, &dims); // 创建一个aclmdlIODims实例
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclmdlGetOutputDims failed, errorCode is %d", ret);
    }
    std::vector<int> dimensions(4); // 创建一个大小为 4 的 vector
    for (size_t i = 0; i < dims.dimCount; ++i)
    {
        dimensions[i] = static_cast<int>(dims.dims[i]); // 将int64_t转换为int并存储到vector中
    }

    aclDataType datatype = aclmdlGetOutputDataType(modelDesc_, 0);

    // copy device output data to host
    aclrtMallocHost(&outHostData, output_length);
    ret = aclrtMemcpy(outHostData, output_length, data, output_length, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("memcpy  failed, errorCode is %d", ret);
        return FAILED;
    }
    outData = reinterpret_cast<float *>(outHostData);

    // 计算总元素数
    size_t total_elements = 1;
    for (int dim : dimensions)
    {
        total_elements *= dim;
    }

    // 确保 output 的大小与模型输出匹配
    output.resize(dimensions[0]); // 假设第一个维度是 batch size
    for (int i = 0; i < dimensions[0]; ++i)
    {
        output[i].resize(total_elements / dimensions[0]);
    }

    // 将 outData 的数据复制到 output 中
    for (int i = 0; i < dimensions[0]; ++i)
    {
        std::memcpy(output[i].data(), outData + i * (total_elements / dimensions[0]), (total_elements / dimensions[0]) * sizeof(float));
    }
    INFO_LOG("FINISH ACNNModel_B::backbone_GetResults");
    return SUCCESS;
}
Result ACNNModel_B::runACNN_B(std::vector<std::vector<float>> &output, cv::Mat &img)
{
    // 根据backbone与head的输入区别进行前处理
    Result ret;
    // 前处理
    ret = backbone_ProcessInput(img);
    if (ret != SUCCESS)
    {
        ERROR_LOG("ProcessInput  failed");
        return FAILED;
    }
    // 推理
    ret = backbone_Inference();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Inference  failed");
        return FAILED;
    }
    // 后处理
    ret = backbone_GetResults(output);
    if (ret != SUCCESS)
    {
        ERROR_LOG("Inference  failed");
        return FAILED;
    }
    return SUCCESS;
}
