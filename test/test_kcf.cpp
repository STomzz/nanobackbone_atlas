#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

    bool HOG = true;
    bool FIXEDWINDOW = true;
    bool MULTISCALE = false;
    bool LAB = false;

    int count_frame = 1;

    // for (int i = 0; i < argc; i++)
    // {
    // 	if (strcmp(argv[i], "hog") == 0)
    // 		HOG = true;
    // 	if (strcmp(argv[i], "fixed_window") == 0)
    // 		FIXEDWINDOW = true;
    // 	if (strcmp(argv[i], "singlescale") == 0)
    // 		MULTISCALE = false;
    // 	if (strcmp(argv[i], "lab") == 0)
    // 	{
    // 		LAB = true;
    // 		HOG = true;
    // 	}
    // 	if (strcmp(argv[i], "gray") == 0)
    // 		HOG = false;
    // }

    // Create KCFTracker object
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    // Tracker_kcf results
    Rect result;

    string inputPath = "../data/tv_tuanliu.mkv";
    string resultsPath = "../results/output_video.mp4";
    Rect bbox_first = Rect(706, 679, 155, 141);

    VideoCapture cap(inputPath);
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }

    // 获取视频帧率和帧尺寸
    double fps = cap.get(CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter video_writer(resultsPath, VideoWriter::fourcc('a', 'v', 'c', '1'), fps, Size(width, height));

    Mat frame;
    if (!cap.read(frame))
    {
        cerr << "Error: Could not read the first frame." << endl;
        return -1;
    }

    tracker.init(bbox_first, frame);

    // 保存第一帧
    rectangle(frame, bbox_first, cv::Scalar(0, 255, 0), 2);
    if (!imwrite("../image/first_frame.jpg", frame))
    {
        cerr << "Error: Could not save the first frame to ../frame/first_frame.jpg" << endl;
        return -1;
    }

    while (cap.read(frame))
    {
        double t1 = getTickCount();
        result = tracker.update(frame);
        float current_pv = tracker.peak_value;
        static cv::Mat temp;
        static cv::Rect temp_roi;

        // 维护高得分帧
        // if (current_pv > 0.75)
        // {
        // 	temp = frame;
        // 	temp_roi = result;
        // 	rectangle(frame, result, Scalar(0, 255, 0), 2);
        // 	imwrite("../frame/temp_frame.jpg", frame);
        // }
        // 低于阈值重新初始化
        // if (current_pv < 0.5 && !temp.empty())
        // {
        // 	printf("start KCF init \n");
        // 	tracker.init(temp_roi, temp);
        // }

        // peak_value添加到图片
        string text = "peak_value : " + to_string(current_pv) + " count : " + to_string(++count_frame) + " Rect(x,y,w,h) : " + to_string(result.x) + "," + to_string(result.y) + "," + to_string(result.width) + "," + to_string(result.height);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        Scalar color(0, 0, 255);
        int thickness = 2;
        Point org(30, 50);
        cv::putText(frame, text, org, fontFace, fontScale, color, thickness);

        rectangle(frame, result, Scalar(0, 255, 0), 2);

        // 实时查看框
        imwrite("../image/debug_frame.jpg", frame);
        video_writer.write(frame);
        // printf("peak_value : %f ", tracker.peak_value);

        double t2 = getTickCount();
        double process_time_ms = (t2 - t1) * 1000 / getTickFrequency();
        double fps_value = getTickFrequency() / (t2 - t1);
        cout << "每帧处理时间: " << process_time_ms << " ms, FPS: " << fps_value << endl;
    }
    // resultsFile.close();
    cap.release();
    video_writer.release();

    return 0;
}