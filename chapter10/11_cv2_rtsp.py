import cv2

# RTSP 流地址
rtsp_url = 'rtsp://admin:1234abcd@192.168.31.53/LiveMedia/ch1/Media1'

# 打开 RTSP 流
cap = cv2.VideoCapture(rtsp_url)

# 检查是否成功打开
if not cap.isOpened():
    print("无法打开 RTSP 流")
    exit()

# 循环读取帧
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    # 显示帧
    cv2.imshow('RTSP Stream', frame)

    # 按键 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
