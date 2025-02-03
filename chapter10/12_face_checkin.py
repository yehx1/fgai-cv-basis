import os
import cv2
import time
import threading
import numpy as np
import pandas as pd
import gradio as gr
import face_recognition
from datetime import datetime



# 初始化数据存储
if not os.path.exists("face_data"):
    os.makedirs("face_data")
if not os.path.exists("实时签到数据.csv"):
    df = pd.DataFrame(columns=["姓名", "时间"])
    df.to_csv("实时签到数据.csv", index=False)

# 加载已注册的人脸数据
def load_registered_faces():
    registered_faces = {}
    for filename in os.listdir("face_data"):
        if filename.endswith(".npy"):
            name = filename.split('.')[0]
            encoding = np.load(f"face_data/{filename}")
            registered_faces[name] = encoding
    return registered_faces

# 加载已注册的人脸数据
registered_faces = load_registered_faces()
# 全局变量保存最新帧
latest_frame = None
attendance_data = pd.read_csv("实时签到数据.csv")  # 用于保存实时签到数据

# 注册人脸
def register_face(name, image):
    global registered_faces
    if image is None:
        return "未提供图像。请重试。", image
    # 将图像转为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测图像中的人脸
    face_locations = face_recognition.face_locations(image_rgb)
    
    if len(face_locations) == 0:
        return "图像中未检测到人脸。请重试。", image
    
    # 计算每个人脸的面积并选择最大的人脸
    face_sizes = [(top, right, bottom, left, (right - left) * (bottom - top)) for (top, right, bottom, left) in face_locations]
    largest_face = max(face_sizes, key=lambda x: x[4])  # 找到面积最大的那个人脸
    largest_top, largest_right, largest_bottom, largest_left, _ = largest_face
    
    # 打印人脸位置
    print(f"人脸位置: {face_locations}")

    # 在图像中绘制人脸框
    for face_location in face_locations:
        top, right, bottom, left = face_location
        # 如果是最大的人脸，使用绿色框，否则使用红色框
        color = (0, 255, 0) if (top, right, bottom, left) == (largest_top, largest_right, largest_bottom, largest_left) else (255, 0, 0)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)  # 绘制框
    
    # 获取最大人脸的编码
    face_encodings = face_recognition.face_encodings(image_rgb, [largest_face[:4]])  # 只提取最大人脸的编码
    if len(face_encodings) > 0:
        # 保存最大人脸编码
        encoding = face_encodings[0]
        name = name.strip()
        if len(name) < 1:
            return "姓名不能为空。请重试。", image
        
        # 如果已存在此名字的人脸，则覆盖
        np.save(f"face_data/{name}.npy", encoding)
        registered_faces[name] = encoding
        
        return f"{name} 注册成功！", image
    else:
        return "图像中未检测到人脸。请重试。", image

# 删除已注册人脸
def delete_registered_face(name):
    global registered_faces
    name = name.strip()
    
    # 如果名字为空或人脸不存在
    if len(name) < 1:
        return "姓名不能为空。请重试。"
    
    if name not in registered_faces:
        return f"未找到 {name} 的人脸数据。"
    
    # 删除文件
    try:
        os.remove(f"face_data/{name}.npy")
        del registered_faces[name]
        return f"{name} 的人脸数据已成功删除。"
    except Exception as e:
        return f"删除人脸失败: {str(e)}"

# 签到函数
def mark_attendance_from_frame(frame):
    global registered_faces, attendance_data
    # 将图像转为RGB格式
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 获取图像中的人脸
    face_encodings = face_recognition.face_encodings(frame_rgb)
    if len(face_encodings) > 0:
        attendance = []
        for encoding in face_encodings:
            # 比对每一张人脸
            for name, registered_encoding in registered_faces.items():
                results = face_recognition.compare_faces([registered_encoding], encoding)
                if results[0]:
                    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_data = pd.concat([pd.DataFrame([{"姓名": name, "时间": time}]), attendance_data], ignore_index=True)

                    attendance_data.to_csv("实时签到数据.csv", index=False)
                    attendance.append(f"{name} 于 {time} 签到成功。")
        if attendance:
            return "\n".join(attendance)
        else:
            return "未检测到已注册的人脸。"
    else:
        return "图像中未检测到人脸。"

# 读取视频流并更新最新帧
def capture_video():
    global latest_frame
    # cap = cv2.VideoCapture(0)  # 使用摄像头捕获视频流
    cap = cv2.VideoCapture("rtsp://admin:1234abcd@192.168.31.53/LiveMedia/ch1/Media1")  # 使用RTSP视频流
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 始终更新为最新的帧
        latest_frame = frame

    cap.release()

# 获取并处理最新帧
def get_latest_frame():
    global latest_frame

    # 模拟数据
    # i = np.random.randint(0, 5, size=1)[0]
    # print(i)
    # if i == 3:
    #     latest_frame = cv2.imread(r"E:\project\python\fgai-cv-basis\chapter10\f1.jpg")

    if latest_frame is not None:
        frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        mark_result = mark_attendance_from_frame(latest_frame)  # 处理最新帧
        return frame_rgb, mark_result
    else:
        empty_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 创建一个黑色的空图像
        return empty_image, "等待视频流..."

# 持续更新视频流（仅处理最新的帧）
def video_feed():
    while True:
        frame, checkin_result = get_latest_frame()  # 获取最新帧和签到信息
        yield gr.update(value=frame), gr.update(value=checkin_result), gr.update(value=attendance_data)  # 更新表格
        time.sleep(1)  # 每秒处理一次（避免过于频繁处理）

# 查询历史记录
def view_history():
    return attendance_data

# 导出历史记录
def export_csv():
    export_path = "历史记录.csv"
    attendance_data.to_csv(export_path, index=False)
    return export_path

# Gradio UI设置
with gr.Blocks() as demo:
    gr.Markdown("## 人脸签到程序")

    with gr.Tab("人脸注册"):
        name_input = gr.Textbox(label="姓名")

        with gr.Row():
            image_input = gr.Image(type="numpy", label="人脸图像", height=300)
            image_output = gr.Image(type="numpy", label="人脸图像", height=300)
        
        with gr.Row():
            register_button = gr.Button("注册人脸")
            delete_button = gr.Button("删除人脸")
        register_output = gr.Textbox(label="日志信息")
        register_button.click(register_face, inputs=[name_input, image_input], outputs=[register_output, image_output])
        delete_button.click(delete_registered_face, inputs=[name_input], outputs=[register_output])
        # 用Dataframe展示已注册人员列表
        registered_names = gr.Dataframe(label="已注册人员列表", headers=["姓名"], col_count=1, value=[[name] for name in registered_faces.keys()])

    with gr.Tab("视频签到"):
        video_output = gr.Image(type="numpy", label="视频流", height=300)
        checkin_output = gr.Textbox(label="签到信息")
        start_button = gr.Button("启动视频签到")
        start_button.click(video_feed, outputs=[video_output, checkin_output, gr.Dataframe(label="已注册人员列表", headers=["姓名", "时间"], col_count=2)], api_name="video_feed")
    
    with gr.Tab("历史记录"):
        history_output = gr.Dataframe(label="已注册人员列表", headers=["姓名", "时间"], col_count=2)
        gr.Button("查看历史").click(view_history, outputs=history_output)
    
    with gr.Tab("导出历史记录"):
        gr.Button("导出CSV").click(export_csv, outputs=gr.File())

# 启动视频流捕捉线程
video_thread = threading.Thread(target=capture_video, daemon=True)
video_thread.start()

# 启动Gradio应用
demo.queue()
demo.launch()
