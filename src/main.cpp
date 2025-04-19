#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <map>
#include <unordered_map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "acl/acl.h"

#include "nanotrack.hpp"
#include "kcftracker.hpp"

// using namespace cv;
// using namespace std;

constexpr bool INFER_ONCE = false;
int low_cf_num = 0;
cv::Rect bbox_first;
std::unordered_map<std::string, cv::Mat> maintain_frame;

void printHelp()
{
    std::cout << "Usage: main <input_video_path>  <output_video_path>  [options]\n"
              << "\t<input_video_path>     \tPath of source input video path.\n"
              << "\t<output_video_path>    \tPath of results(.mp4) to save.\n"
              << "\t<input_video Rect.x>    \tinput video bbox_first.x\n"
              << "\t<input_video Rect.y>    \tPnput video bbox_first.y\n"
              << "\t<input_video Rect.width>    \tnput video bbox_first.width\n"
              << "\t<input_video Rect.height>    \tnput video bbox_first.height\n"
              << "Options:\n"
              << "\t--help,-h        \tPrint usage information and exit.\n"
              << std::endl;
}
void draw_text(cv::Mat &frame, std::string text)
{
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    cv::Scalar color(0, 0, 255);
    int thickness = 2;
    cv::Point org(30, 50);
    cv::putText(frame, text, org, fontFace, fontScale, color, thickness);
}

Result ACNNInit(int32_t &deviceId, aclrtContext &context, aclrtStream &stream)
{
    // init acl resource
    const char *aclConfigPath = "";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclInit failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtSetDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateContext(&context, deviceId);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtCreateContext failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtCreateStream failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result ACNNDeInit(int32_t &deviceId, aclrtContext &context, aclrtStream &stream)
{
    aclError ret;

    if (stream != nullptr)
    {
        ret = aclrtDestroyStream(stream);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("aclrtDestroyStream failed, errorCode is %d", ret);
        }
        stream = nullptr;
    }

    if (context != nullptr)
    {
        ret = aclrtDestroyContext(context);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("aclrtDestroyContext failed, errorCode is %d", ret);
        }
        context = nullptr;
    }

    ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclrtResetDevice failed, errorCode is %d", ret);
    }

    ret = aclFinalize();
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("aclFinalize failed, errorCode is %d", ret);
    }
    return SUCCESS;
}

bool fileExists(const std::string &filePath)
{
    std::ifstream file(filePath);
    return file.good();
}

int main(int argc, char **argv)
{
    /*-------------------------------------------------参数解析-------------------------------------------------*/
    if (argc < 7)
    {
        printHelp();
        return -1;
    }
    for (int i = 7; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            printHelp();
            return -1;
        }
        else
        {
            printHelp();
            std::cout << "\n"
                      << "Unsupported options: " << argv[i] << std::endl;
            return -1;
        }
    }
    std::string video_path = *(argv + 1);
    std::string video_out_path = *(argv + 2);
    bbox_first.x = std::stoi(*(argv + 3));
    bbox_first.y = std::stoi(*(argv + 4));
    bbox_first.width = std::stoi(*(argv + 5));
    bbox_first.height = std::stoi(*(argv + 6));
    //(706, 679, 155, 141);

    /*---------------------------------------------------ACNNInit---------------------------------------------------*/
    int32_t deviceId_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    Result ret = ACNNInit(deviceId_, context_, stream_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("ACNNInit  failed");
        return FAILED;
    }
    else
    {
        INFO_LOG("ACNNInit SUCCES");
    }

    /*---------------------------------------------------Nanotrack 加载模型---------------------------------------------------*/

    const char *T_backbone_model = "../model/nanotrack_backbone_127.om";
    const char *X_backbone_model = "../model/nanotrack_backbone_255.om";
    const char *head_model = "../model/nanotrack_head.om";
    if (!fileExists(T_backbone_model))
    {
        std::cerr << "Model file does not exist: " << T_backbone_model << std::endl;
        return 1;
    }
    if (!fileExists(X_backbone_model))
    {
        std::cerr << "Model file does not exist: " << X_backbone_model << std::endl;
        return 1;
    }

    if (!fileExists(head_model))
    {
        std::cerr << "Model file does not exist: " << head_model << std::endl;
        return 1;
    }

    NanoTrack nanotrack(T_backbone_model, X_backbone_model, head_model);
    INFO_LOG(" nanotrack create SUCCES");
    nanotrack.initsource();
    INFO_LOG(" nanotrack initsource SUCCES");

    /*------------------------------------------------------视频帧初始化-------------------------------------------------------*/
    // const string video_path = "../data/girl_dance.mp4";
    int frame_count = 1;
    // const string video_path = "../data/tv_tuanliu.mkv";
    cv::VideoCapture cap(video_path);
    // 检查视频是否成功打开
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video.\n";
        return -1;
    }
    // 获取视频帧率和帧尺寸
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 创建视频写入器
    // VideoWriter video_writer("../results/output_video.mp4", VideoWriter::fourcc('a', 'v', 'c', '1'), fps, Size(width, height));
    cv::VideoWriter video_writer(video_out_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(width, height));
    // 读取一帧初始帧
    cv::Mat frame;
    if (!cap.read(frame))
    {
        std::cerr << "Error: Could not read the first frame." << std::endl;
        return -1;
    }
    // 维护第一帧模板

    maintain_frame["first"] = frame;

    // Rect bbox_first(275, 149, 62, 60);
    // Rect bbox_first(706, 679, 155, 141);
    // 绘制边界框
    cv::rectangle(frame, bbox_first, cv::Scalar(0, 255, 0), 2);
    // 保存第一帧到本地
    std::string output_path = "../results/first_frame_with_bbox.jpg";
    if (!imwrite(output_path, frame))
    {
        std::cerr << "Error: Could not save the first frame to " << output_path << std::endl;
        return -1;
    }
    std::cout << "First frame with bounding box saved to " << output_path << std::endl;
    nanotrack.init(frame, bbox_first);
    INFO_LOG(" nanotrack.init SUCCES");

    // std::vector<float> video_bbox;
    cv::Rect video_bbox;
    /*------------------------------------------------------视频帧初始化-------------------------------------------------------*/

    /*------------------------------------------------------视频帧追踪-------------------------------------------------------*/
    while (true)
    {
        if (!cap.read(frame))
        {
            std::cout << "End of video or unable to read frame." << std::endl;
            break;
        }

        double t1 = cv::getTickCount();
        // 启动追踪
        video_bbox = nanotrack.track(frame);
        double t2 = cv::getTickCount();
        double process_time_ms = (t2 - t1) * 1000 / cv::getTickFrequency();
        double fps_value = cv::getTickFrequency() / (t2 - t1);
        std::cout << "每帧处理时间: " << process_time_ms << " ms, FPS: " << fps_value << std::endl;
        // cv::Rect video_bbox(video_bbox[0], video_bbox[1], video_bbox[2], video_bbox[3]);
        if (video_bbox.x < 0 || video_bbox.y < 0 || video_bbox.x + video_bbox.width > frame.cols || video_bbox.y + video_bbox.height > frame.rows)
        {
            std::cerr << "Error: ROI is out of frame bounds!" << std::endl;
            return -1;
        }
        //  绘制边界框
        cv::rectangle(frame, video_bbox, cv::Scalar(0, 255, 0), 3);
        INFO_LOG(" Rectangle frame SUCCES");
        // debug frame
        cv::imwrite("../image/debug_frame.jpg", frame);

        // 写入视频
        video_writer.write(frame);
        frame_count++;
        std::cout << "now frame  is " << frame_count << std::endl;
    }
    // 释放资源
    video_writer.release();
    cap.release();

    /*------------------------------------------------------视频帧追踪-------------------------------------------------------*/
    ret = ACNNDeInit(deviceId_, context_, stream_);
    if (ret != SUCCESS)
    {
        ERROR_LOG("ACNNDeInit  failed");
        return FAILED;
    }
    else
    {
        INFO_LOG("ACNNDeInit SUCCES");
    }

    return 0;
}
