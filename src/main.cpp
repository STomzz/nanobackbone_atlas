#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

// 四维张量类型定义 [batch][channel][height][width]
using Tensor4D = vector<vector<vector<vector<double>>>>;

// 单通道二维卷积函数（步长1，填充0）
vector<vector<double>> conv2d_single_channel(
    const vector<vector<double>> &input,
    const vector<vector<double>> &kernel)
{
    const int input_h = input.size();
    const int input_w = input[0].size();
    const int kernel_h = kernel.size();
    const int kernel_w = kernel[0].size();

    // 计算输出尺寸
    const int output_h = input_h - kernel_h + 1;
    const int output_w = input_w - kernel_w + 1;

    // 执行卷积（无填充，步长1）
    vector<vector<double>> output(output_h, vector<double>(output_w, 0.0));
    for (int i = 0; i < output_h; ++i)
    {
        for (int j = 0; j < output_w; ++j)
        {
            double sum = 0.0;
            for (int m = 0; m < kernel_h; ++m)
            {
                for (int n = 0; n < kernel_w; ++n)
                {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

// 主卷积函数
Tensor4D special_conv(
    const Tensor4D &input,   // [1, 96, 16, 16]
    const Tensor4D &kernels) // [1, 96, 8, 8]
{
    // 参数校验
    if (input.empty() || kernels.empty() ||
        input[0].size() != 96 || kernels[0].size() != 96)
        throw invalid_argument("Invalid input shape");

    // 初始化输出张量 [1, 96, 9, 9]
    Tensor4D output(1, vector<vector<vector<double>>>(
                           96, vector<vector<double>>(9, vector<double>(9, 0.0))));

    // 对每个卷积核进行处理
    for (int k = 0; k < 96; ++k)
    { // 遍历96个卷积核
        // 获取当前卷积核 [8x8]
        const auto &kernel = kernels[0][k];

        // 初始化累加器
        vector<vector<double>> accumulator(9, vector<double>(9, 0.0));

        // 遍历所有输入通道
        for (int c = 0; c < 96; ++c)
        { // 遍历96个输入通道
            // 获取输入通道 [16x16]
            const auto &input_ch = input[0][c];

            // 执行卷积操作
            auto conv_result = conv2d_single_channel(input_ch, kernel);

            // 累加结果
            for (int i = 0; i < 9; ++i)
            {
                for (int j = 0; j < 9; ++j)
                {
                    accumulator[i][j] += conv_result[i][j];
                }
            }
        }

        // 存储最终结果
        output[0][k] = accumulator;
    }

    return output;
}

// 打印9x9矩阵的辅助函数
void print_9x9(const vector<vector<double>> &mat)
{
    for (const auto &row : mat)
    {
        for (double val : row)
        {
            printf("%8.1f", val);
        }
        cout << endl;
    }
}

int main()
{
    // 创建测试数据（全1矩阵）
    Tensor4D input(1, vector<vector<vector<double>>>(
                          96, vector<vector<double>>(16, vector<double>(16, 1.0))));

    Tensor4D kernels(1, vector<vector<vector<double>>>(
                            96, vector<vector<double>>(8, vector<double>(8, 1.0))));

    // 执行卷积
    auto result = special_conv(input, kernels);

    // 验证第一个卷积核的输出结果
    cout << "First kernel output (should be 96*64=6144):" << endl;
    print_9x9(result[0][0]); // 所有位置值应为6144.0

    return 0;
}