#include <iostream>
#include <cstring> // for memset

// 四维张量指针结构
struct Tensor4D
{
    float *data; // 数据指针[N][C][H][W]
    int dims[4]; // 各维度大小
};

// 卷积核维度计算
inline int calc_offset(int w, int h, int channel, int width)
{
    return channel * (h * width) + h * width + w;
}

// 单次卷积操作
void single_convolution(const float *input,  // 输入数据指针[16][16]
                        const float *kernel, // 卷积核指针[8][8]
                        float *output,       // 输出数据指针[9][9]
                        int input_size = 16,
                        int kernel_size = 8)
{
    const int output_size = input_size - kernel_size + 1;

    // 遍历输出每个位置
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            float sum = 0.0f;

            // 卷积核滑动窗口
            for (int m = 0; m < kernel_size; ++m)
            {
                for (int n = 0; n < kernel_size; ++n)
                {
                    // 计算输入位置
                    const int x = i + m;
                    const int y = j + n;

                    // 累加乘积
                    sum += input[x * input_size + y] *
                           kernel[m * kernel_size + n];
                }
            }
            output[i * output_size + j] = sum;
        }
    }
}

// 主卷积函数
void grouped_convolution(const Tensor4D &input,   // [1,96,16,16]
                         const Tensor4D &kernels, // [1,96,8,8]
                         Tensor4D &output)        // [1,96,9,9]
{
    // 参数校验
    if (input.dims[0] != 1 || kernels.dims[0] != 1)
        throw std::invalid_argument("Batch size must be 1");
    if (input.dims[1] != 96 || kernels.dims[1] != 96)
        throw std::invalid_argument("Channel number must be 96");
    if (input.dims[2] != 16 || input.dims[3] != 16)
        throw std::invalid_argument("Input must be 16x16");
    if (kernels.dims[2] != 8 || kernels.dims[3] != 8)
        throw std::invalid_argument("Kernel must be 8x8");

    // 预计算各维度步长
    const int input_channel_size = 16 * 16;
    const int kernel_channel_size = 8 * 8;
    const int output_channel_size = 9 * 9;

    // 遍历每个输出通道（每个卷积核）
    for (int out_c = 0; out_c < 96; ++out_c)
    {
        // 当前卷积核指针
        const float *curr_kernel = kernels.data +
                                   out_c * kernel_channel_size;

        // 当前输出通道指针
        float *output_channel = output.data +
                                out_c * output_channel_size;

        // 初始化输出为0
        memset(output_channel, 0, output_channel_size * sizeof(float));

        // 遍历所有输入通道
        for (int in_c = 0; in_c < 96; ++in_c)
        {
            // 临时存储单次卷积结果
            float conv_result[9 * 9] = {0};

            // 输入通道指针
            const float *input_channel = input.data +
                                         in_c * input_channel_size;

            // 执行单通道卷积
            single_convolution(input_channel, curr_kernel, conv_result);

            // 累加到输出通道
            for (int i = 0; i < 81; ++i)
            { // 9x9=81
                output_channel[i] += conv_result[i];
            }
        }
    }
}

// 示例用法
int main()
{
    // 初始化张量 (实际使用时应为真实数据)
    Tensor4D input = {
        .data = new float[1 * 96 * 16 * 16](), // 初始化为0
        .dims = {1, 96, 16, 16}};

    Tensor4D kernels = {
        .data = new float[1 * 96 * 8 * 8](),
        .dims = {1, 96, 8, 8}};

    Tensor4D output = {
        .data = new float[1 * 96 * 9 * 9](),
        .dims = {1, 96, 9, 9}};

    // 填充测试数据（全1）
    std::fill_n(input.data, 1 * 96 * 16 * 16, 1.0f);
    std::fill_n(kernels.data, 1 * 96 * 8 * 8, 1.0f);

    // 执行卷积
    try
    {
        grouped_convolution(input, kernels, output);

        // 验证结果（第一个输出通道应全为8x8x96=6144）
        std::cout << "First output channel (3,3): "
                  << output.data[3 * 9 + 3] << std::endl; // 应输出6144
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // 释放内存
    delete[] input.data;
    delete[] kernels.data;
    delete[] output.data;

    return 0;
}