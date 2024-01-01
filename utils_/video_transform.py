# 把视频转换为裁剪好的avi格式
import cv2

video = 'tests/test.mp4'  # 视频地址
cap = cv2.VideoCapture(video)
video_writer = cv2.VideoWriter('test_4_1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))  # 转换格式
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640, 480))
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        video_writer.release()
        break
