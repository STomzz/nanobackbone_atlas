import cv2

def select_roi(video_path):
    # 初始化视频捕获对象
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 读取首帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频首帧")
        return

    # 创建窗口并显示首帧
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select ROI", frame)

    # 交互式ROI选择
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # 解包坐标 (x,y,width,height) => (711, 680, 156, 127)
    x, y, w, h = roi
    print(roi)
    coordinates = f"x={x}, y={y}, width={w}, height={h}"

    # 释放资源
    cap.release()

if __name__ == "__main__":
    video_path = "../video/original/input.mkv"    # 输入视频路径
    select_roi(video_path)