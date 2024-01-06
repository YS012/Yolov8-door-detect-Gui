import ctypes

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor, QCursor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, Qt
from pykinect2 import PyKinectRuntime, PyKinectV2

from ui.CustomMessageBox import MessageBox
from ui.mainwindow import Ui_MainWindow
from ui.UIFunctions import *
from collections import defaultdict
from pathlib import Path
from utils_.capnums import Camera
from utils_.rtsp_win import Window
from utils_.id_win import id_Window
import threading
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import sys
import cv2

import datetime

from matplotlib.colors import LinearSegmentedColormap
import matplotlib

import os
import supervision as sv
from ultralytic import YOLO
from ultralytic.yolo.data.dataloaders.stream_loaders import LoadStreams
from ultralytic.yolo.engine.predictor import BasePredictor
from ultralytic.yolo.utils import DEFAULT_CFG, SETTINGS, callbacks
from ultralytic.yolo.utils.torch_utils import smart_inference_mode
from ultralytic.yolo.utils.files import increment_path
from ultralytic.yolo.utils.checks import check_imshow
from ultralytic.yolo.cfg import get_cfg
from collections import deque

matplotlib.use('TkAgg')
# plt显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 创建一个渐变色
gradient = LinearSegmentedColormap.from_list(
    'gradient', [(0, 0, 0), (233, 156, 105)], N=256)

video_id_count = 0
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]  # 颜色板
dic_for_drawing_trails = {}

myapp_id = "my_app"
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myapp_id)


def compute_color_for_labels(label):
    """
    设置不同类别的固定颜色
    """
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# 绘制轨迹
def draw_trail(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    try:
        for key in list(dic_for_drawing_trails):
            if key not in identities:
                dic_for_drawing_trails.pop(key)
    except:
        pass

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # 获取锚框boundingbox中心点
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # 获取目标ID
        id = int(identities[i]) if identities is not None else 0
        # 创建新的缓冲区
        if id not in dic_for_drawing_trails:
            dic_for_drawing_trails[id] = deque(maxlen=64)
        try:
            color = compute_color_for_labels(object_id[i])
        except:
            continue

        dic_for_drawing_trails[id].appendleft(center)
        # 绘制轨迹
        for i in range(1, len(dic_for_drawing_trails[id])):

            if dic_for_drawing_trails[id][i - 1] is None or dic_for_drawing_trails[id][i] is None:
                continue
            # 轨迹动态粗细
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            img = cv2.line(img, dic_for_drawing_trails[id][i - 1], dic_for_drawing_trails[id][i], color, thickness)
    return img


class YoloPredictor(BasePredictor, QObject):
    yolo2main_trail_img = Signal(np.ndarray)  # 轨迹图像信号
    yolo2main_box_img = Signal(np.ndarray)  # 绘制了标签与锚框的图像的信号
    yolo2main_status_msg = Signal(str)  # 检测/暂停/停止/测试完成等信号
    yolo2main_fps = Signal(str)  # fps
    yolo2main_labels = Signal(dict)  # 检测到的目标结果（每个类别的数量）
    yolo2main_progress = Signal(int)  # 进度条
    yolo2main_class_num = Signal(int)  # 当前帧类别数
    yolo2main_target_num = Signal(int)  # 当前帧目标数

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()

        QObject.__init__(self)
        try:
            self.args = get_cfg(cfg, overrides)
        except:
            pass
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI args
        self.used_model_name = None  # 使用过的检测模型名称
        self.new_model_name = None  # 新更改的模型
        self.source = ''  # 输入源str
        self.stop_dtc = False  # 终止bool
        self.continue_dtc = True  # 暂停bool
        self.show_graph = False  # 折线图展示bool
        self.save_res = False  # 保存MP4
        self.save_txt = False  # 保存txt
        self.show_labels = True  # 显示图像标签bool
        self.iou_thres = 0.45  # iou
        self.conf_thres = 0.25  # conf
        self.speed_thres = 10  # delay, ms
        self.progress_value = 0  # 进度条的值
        self.kinect_selected = 0  # kinect摄像头被选定标志

        self.run_started = 0
        self.loop_flag = 0
        self.stop_thread = 0
        self.X_quit = 0
        self.lock_id = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)

    # 单目标跟踪
    def single_object_tracking(self, detections, img_box, org_2, store_xyxy_for_id):
        for xyxy, id in zip(detections.xyxy, detections.tracker_id):
            store_xyxy_for_id[id] = xyxy
            mask = np.zeros_like(img_box)
        try:
            if self.lock_id not in detections.tracker_id:
                cv2.destroyAllWindows()
                self.lock_id = None
            x1, y1, x2, y2 = int(store_xyxy_for_id[self.lock_id][0]), int(store_xyxy_for_id[self.lock_id][1]), int(
                store_xyxy_for_id[self.lock_id][2]), int(store_xyxy_for_id[self.lock_id][3])
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            result_mask = cv2.bitwise_and(org_2, mask)
            result_cropped = result_mask[y1:y2, x1:x2]
            result_cropped = cv2.resize(result_cropped, (256, 256))

            return result_cropped

        except:
            cv2.destroyAllWindows()

    # 点击开始检测按钮后的检测事件
    @smart_inference_mode()  # 一个修饰器，用来开启检测模式：如果torch>=1.9.0，则执行torch.inference_mode()，否则执行torch.no_grad()
    def run(self):
        # try:
        LoadStreams.capture = None
        # self.sources = 0
        self.run_started = 1

        global video_id_count

        self.yolo2main_status_msg.emit('正在加载模型...')

        # 检查保存路径
        if self.save_txt:
            if not os.path.exists('labels'):
                os.mkdir('labels')
        if self.save_res:
            if not os.path.exists('pred_result'):
                os.mkdir('pred_result')

        count = 0  # 拿来参与算FPS的计数变量
        start_time = time.time()  # 拿来算FPS的计数变量
        start_time_graph = time.time()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # box_annotator = sv.BoxAnnotator(
        #     thickness=2,
        #     text_thickness=1,
        #     text_scale=0.5
        # )
        START = sv.Point(0, 320)
        END = sv.Point(640, 320)
        line_counter = sv.LineZone(start=START, end=END)
        line_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.6
        )
        cv_start_location_x = 0
        cv_start_location_y = 0

        if self.continue_dtc:  # 暂停与继续的切换

            # try:
            #     out.release()
            # except:
            #     pass
            if self.used_model_name != self.new_model_name:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name
            model = YOLO(self.new_model_name)
            iter_model = iter(
                model.track(source=self.source, show=False, stream=True, iou=self.iou_thres, conf=self.conf_thres))
            self.yolo2main_status_msg.emit('检测中...')
            kinect = None
            if self.kinect_selected:
                kinect = PyKinectRuntime.PyKinectRuntime(
                    PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

            flag_save_video = 1  # 拿来保存视频的flag，免得在后面的循环里面重复执行cv2.VideoWriter()函数
            t_list_for_x_axis_in_graph_display = []
            result_list_for_y_axis_in_graph_display = []
            current_time = 0

            sum_of_count = 0

            if 'mp4' in self.source or 'avi' in self.source or 'mkv' in self.source or 'flv' in self.source or 'mov' in self.source:
                cap = cv2.VideoCapture(self.source)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()

            store_xyxy_for_id = {}
            # if self.kinect_selected:
            #     frame_count = 0

            while True:
                try:
                    if self.continue_dtc:
                        result = next(iter_model)  # 这里是检测的核心，每次循环都会检测一帧图像,可以自行打印result看看里面有哪些key可以用
                        img_trail = result.orig_img
                        # org = np.copy(img_trail)
                        org_2 = np.copy(img_trail)

                        # if self.kinect_selected and kinect is not None:
                        #     frame_count += 1
                        #     if frame_count > 8:
                        #         frame_depth = kinect.depth_frame_data
                        #         key_points = result.keypoints
                        #         if key_points:
                        #             key_points_xy = result.keypoints.xy[0]
                        #             for point in key_points_xy:
                        #                 x = int(point[0])
                        #                 y = int(point[1])
                        #                 if (x + y) != 0:
                        #                     print(f'center:({x, y}) ', 'depth',
                        #                           frame_depth[y * kinect.depth_frame_desc.Width + x])
                        #         frame_count = 0

                        class_num_arr = []
                        detections = sv.Detections.from_ultralytics(result)
                        for each in detections.class_id:
                            if each not in class_num_arr:
                                class_num_arr.append(each)
                        class_num = len(class_num_arr)
                        try:
                            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                        except:
                            pass
                        labels = [
                            f"OBJECT-ID: {tracker_id} CLASS: {model.model.names[class_id]} CF: {confidence:0.2f}"
                            for _, _, confidence, class_id, tracker_id in detections
                        ]
                        labels2 = [
                            f"目标ID: {tracker_id} 目标类别: {model.model.names[class_id]} 置信度: {confidence:0.2f}"
                            for _, _, confidence, class_id, tracker_id in detections
                        ]
                        # 存储labels里的信息
                        if self.save_txt:
                            with open('labels/result.txt', 'a') as f:
                                f.write('当前时刻屏幕信息:' + str(
                                    labels2) + f'检测时间: {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}' + f' 通过的目标总数: {sum_of_count}')
                                f.write('\n')
                        id = detections.tracker_id
                        xyxy = detections.xyxy
                        if self.show_labels:
                            # img_box = box_annotator.annotate(scene=org, detections=detections, labels=labels)
                            img_box = result.plot()
                        elif not self.show_labels:
                            img_box = org_2
                        temp_sum = sum_of_count
                        line_counter.trigger(detections=detections)
                        # line_annotator.annotate(frame=img_box, line_counter=line_counter)
                        sum_of_count = line_counter.in_count + line_counter.out_count
                        identities = id
                        height, width, _ = img_box.shape
                        img_trail = np.zeros((height, width, 3), dtype='uint8')
                        grid_color = (255, 255, 255)
                        line_width = 1
                        grid_size = 100
                        for y in range(0, height, grid_size):
                            cv2.line(img_trail, (0, y), (width, y), grid_color, line_width)
                        for x in range(0, width, grid_size):
                            cv2.line(img_trail, (x, 0), (x, height), grid_color, line_width)
                        draw_trail(img_trail, xyxy, model.model.names, id, identities)
                        height, width, _ = img_box.shape
                        if self.save_res and flag_save_video:
                            out = cv2.VideoWriter(f'pred_result/video_result_{video_id_count}.avi', fourcc, 25,
                                                  (width, height), True)  # 保存检测视频的路径
                            flag_save_video = 0
                        if self.stop_dtc:
                            if self.save_res:
                                out.release()
                                video_id_count += 1
                            self.source = None
                            self.yolo2main_status_msg.emit('检测终止')
                            LoadStreams.capture = 'release'  # 这里是为了终止使用摄像头检测函数的线程，改了yolo源码
                            break
                        try:
                            current_time = time.time() - start_time_graph
                            if sum_of_count not in result_list_for_y_axis_in_graph_display:
                                t_list_for_x_axis_in_graph_display.append(datetime.datetime.now())
                                result_list_for_y_axis_in_graph_display.append(sum_of_count)

                            self.yolo2main_trail_img.emit(img_trail)
                            if self.show_graph:
                                self.yolo2main_trail_img.emit(img_trail)
                            time.sleep(0.0)  # 缓冲
                            self.yolo2main_box_img.emit(img_box)

                            # 进度条
                            try:
                                self.progress_value = int(count / total_frames * 1000)
                                self.yolo2main_progress.emit(self.progress_value)
                            except:
                                pass

                            # 抠锚框里的图
                            if self.lock_id is not None:
                                self.lock_id = int(self.lock_id)
                                try:

                                    result_cropped = self.single_object_tracking(detections, img_box, org_2,
                                                                                 store_xyxy_for_id)
                                    # print(result_cropped)
                                    cv2.imshow(f'OBJECT-ID:{self.lock_id}', result_cropped)
                                    cv2.moveWindow(f'OBJECT-ID:{self.lock_id}', 0, 0)
                                    # press esc to quit
                                    if cv2.waitKey(5) & 0xFF == 27:
                                        self.lock_id = None
                                        cv2.destroyAllWindows()
                                except:
                                    cv2.destroyAllWindows()
                                    pass
                            else:
                                try:

                                    cv2.destroyAllWindows()
                                except:
                                    pass

                            if self.save_res:
                                out.write(img_box)
                            self.yolo2main_class_num.emit(class_num)
                            self.yolo2main_target_num.emit(len(detections.tracker_id))
                        except:
                            pass
                        count += 1

                        if count % 3 == 0 and count >= 3:  # 计算FPS
                            self.yolo2main_fps.emit(str(int(3 / (time.time() - start_time))))
                            start_time = time.time()
                    else:
                        if self.stop_dtc:
                            if self.save_res:
                                out.release()
                                video_id_count += 1
                            self.source = None
                            self.yolo2main_status_msg.emit('检测终止')

                            break

                # 检测截止（本地文件检测）
                except StopIteration:
                    if self.save_res:
                        out.release()
                        video_id_count += 1
                        print('writing complete')
                    self.yolo2main_status_msg.emit('检测完成')
                    self.yolo2main_progress.emit(1000)
                    cv2.destroyAllWindows()
                    break
            try:
                out.release()
            except:
                pass

    # except Exception as e:
    # pass
    # print(e)
    # self.yolo2main_status_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_MainWindow):
    main2yolo_begin_sgl = Signal()  # 主窗口向yolo实例发送执行信号

    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        self.setupUi(self)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)
        UIFuncitons.uiDefinitions(self)

        UIFuncitons.shadow_style(self, self.Class_QF, QColor(0, 205, 102))
        UIFuncitons.shadow_style(self, self.Target_QF, QColor(123, 104, 238))
        UIFuncitons.shadow_style(self, self.Fps_QF, QColor(0, 205, 102))
        UIFuncitons.shadow_style(self, self.Model_QF, QColor(123, 104, 238))

        self.model_box.clear()
        self.pt_list = os.listdir('./weights')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt') or file.endswith('.engine')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./weights/' + x))  # 按文件大小排序
        self.model_box.clear()
        self.model_box.addItems(self.pt_list)
        self.Qtimer_ModelBox = QTimer(self)  # 计时器：每2秒监视模型文件更改一次
        self.Qtimer_ModelBox.timeout.connect(self.ModelBoxRefre)
        self.Qtimer_ModelBox.start(2000)

        # Yolo-v8 thread
        self.yolo_predict = YoloPredictor()  # 实例化yolo检测
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./weights/%s" % self.select_model
        self.yolo_thread = QThread()
        self.yolo_predict.yolo2main_trail_img.connect(lambda x: self.show_image(x, self.pre_video))
        self.yolo_predict.yolo2main_box_img.connect(lambda x: self.show_image(x, self.res_video))
        self.yolo_predict.yolo2main_status_msg.connect(lambda x: self.show_status(x))
        self.yolo_predict.yolo2main_fps.connect(lambda x: self.fps_label.setText(x))

        self.yolo_predict.yolo2main_class_num.connect(lambda x: self.Class_num.setText(str(x)))
        self.yolo_predict.yolo2main_target_num.connect(lambda x: self.Target_num.setText(str(x)))
        self.yolo_predict.yolo2main_progress.connect(lambda x: self.progress_bar.setValue(x))
        self.main2yolo_begin_sgl.connect(self.yolo_predict.run)
        self.yolo_predict.moveToThread(self.yolo_thread)

        # 模型参数
        self.model_box.currentTextChanged.connect(self.change_model)
        self.iou_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'iou_spinbox'))  # iou box
        self.iou_slider.valueChanged.connect(lambda x: self.change_val(x, 'iou_slider'))  # iou scroll bar
        self.conf_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'conf_spinbox'))  # conf box
        self.conf_slider.valueChanged.connect(lambda x: self.change_val(x, 'conf_slider'))  # conf scroll bar
        self.speed_spinbox.valueChanged.connect(lambda x: self.change_val(x, 'speed_spinbox'))  # speed box
        self.speed_slider.valueChanged.connect(lambda x: self.change_val(x, 'speed_slider'))  # speed scroll bar

        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.Model_name.setText(self.select_model)

        self.src_file_button.clicked.connect(self.open_src_file)
        self.src_cam_button.clicked.connect(self.camera_select)
        self.src_rtsp_button.clicked.connect(self.rtsp_seletction)
        self.src_lock_button.clicked.connect(self.lock_id_selection)

        self.run_button.clicked.connect(self.run_or_continue)
        self.stop_button.clicked.connect(self.stop)

        self.save_res_button.toggled.connect(self.is_save_res)
        self.save_txt_button.toggled.connect(self.is_save_txt)
        self.show_labels_checkbox.toggled.connect(self.is_show_labels)
        self.ToggleBotton.clicked.connect(lambda: UIFuncitons.toggleMenu(self, True))
        self.settings_button.clicked.connect(lambda: UIFuncitons.settingBox(self, True))

        self.load_config()

    # 主窗口显示轨迹图像和检测图像
    @staticmethod
    def show_image(img_src, label):
        try:
            if len(img_src.shape) == 3:
                ih, iw, _ = img_src.shape
            if len(img_src.shape) == 2:
                ih, iw = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def set_lock_id(self, lock_id):
        self.yolo_predict.lock_id = None
        self.yolo_predict.lock_id = lock_id
        new_config = {"id": lock_id}
        new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
        with open('config/id.json', 'w', encoding='utf-8') as f:
            f.write(new_json)
        self.show_status('加载ID:{}'.format(lock_id))
        self.id_window.close()

    # 控制开始|暂停
    def run_or_continue(self):
        if self.yolo_predict.source == '' or self.yolo_predict.source == None:
            self.show_status('请在检测前选择输入源...')
            self.run_button.setChecked(False)
        else:
            self.yolo_predict.stop_dtc = False
            if self.run_button.isChecked():
                self.run_button.setChecked(True)
                self.save_txt_button.setEnabled(False)
                self.save_res_button.setEnabled(False)
                self.conf_slider.setEnabled(False)
                self.iou_slider.setEnabled(False)
                self.speed_slider.setEnabled(False)

                self.show_status('检测中...')
                if '0' in self.yolo_predict.source or 'rtsp' in self.yolo_predict.source:
                    self.progress_bar.setFormat('实时视频流检测中...')
                if 'avi' in self.yolo_predict.source or 'mp4' in self.yolo_predict.source:
                    self.progress_bar.setFormat("当前检测进度:%p%")
                self.yolo_predict.continue_dtc = True
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.main2yolo_begin_sgl.emit()

            else:
                self.yolo_predict.continue_dtc = False
                self.show_status("暂停...")

                self.run_button.setChecked(False)

    def show_status(self, msg):
        self.status_bar.setText(msg)
        if msg == 'Detection completed' or msg == '检测完成':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 终止线程
            self.yolo_predict.kinect_selected = 0
        elif msg == 'Detection terminated!' or msg == '检测终止':
            self.save_res_button.setEnabled(True)
            self.save_txt_button.setEnabled(True)
            self.run_button.setChecked(False)
            self.progress_bar.setValue(0)
            if self.yolo_thread.isRunning():
                self.yolo_thread.quit()  # 终止线程
            self.pre_video.clear()
            self.res_video.clear()
            self.Class_num.setText('--')
            self.Target_num.setText('--')
            self.fps_label.setText('--')
            self.yolo_predict.kinect_selected = 0

    def open_src_file(self):
        config_file = 'config/fold.json'
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold,
                                              "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png)")
        if name:
            self.yolo_predict.source = name
            self.show_status('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()
            # 选择摄像头

    def camera_select(self):
        # try:
        pos = QCursor.pos()
        self.stop()
        # 获取本地摄像头数量
        _, cams = Camera().get_cam_num()
        popMenu = QMenu()
        popMenu.setFixedWidth(self.src_cam_button.width())
        popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 20px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 212, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 250, 200,50);}
                                            ''')

        for cam in cams:
            exec("action_%s = QAction('%s 号摄像头')" % (cam, cam))
            exec("popMenu.addAction(action_%s)" % cam)
        action = popMenu.exec(pos)
        if action:
            str_temp = ''
            selected_stream_source = str_temp.join(filter(str.isdigit, action.text()))  # 获取摄像头号，去除非数字字符
            if selected_stream_source == str((len(cams) - 1)):
                self.yolo_predict.kinect_selected = 1
            else:
                self.yolo_predict.kinect_selected = 0
            self.yolo_predict.source = selected_stream_source
            self.show_status('摄像头设备:{}'.format(action.text()))

    # except Exception as e:
    # self.show_status('%s' % e)

    # 选择rtsp
    def rtsp_seletction(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin@10.98.43.107:8554/live"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    # 加载RTSP
    def load_rtsp(self, ip):
        # try:
        self.stop()
        MessageBox(
            self.close_button, title='提示', text='加载 rtsp...', time=1000, auto=True).exec()
        self.yolo_predict.source = ip
        new_config = {"ip": ip}
        new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
        with open('config/ip.json', 'w', encoding='utf-8') as f:
            f.write(new_json)
        self.show_status('加载rtsp地址:{}'.format(ip))
        self.rtsp_window.close()

    # except Exception as e:
    # self.show_status('%s' % e)

    def lock_id_selection(self):
        self.yolo_predict.lock_id = None
        self.id_window = id_Window()
        config_file = 'config/id.json'
        if not os.path.exists(config_file):
            id = ""
            new_config = {"id": id}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            id = config['id']
        self.id_window.idEdit.setText(id)
        self.id_window.show()
        self.id_window.idButton.clicked.connect(lambda: self.set_lock_id(self.id_window.idEdit.text()))

    # 保存提示（MP4）
    def is_save_res(self):
        if self.save_res_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('提示: 监测结果不会被保存')
            self.yolo_predict.save_res = False
        elif self.save_res_button.checkState() == Qt.CheckState.Checked:
            self.show_status('提示: 监测结果将会被保存')
            self.yolo_predict.save_res = True

    # 保存提示（txt）
    def is_save_txt(self):
        if self.save_txt_button.checkState() == Qt.CheckState.Unchecked:
            self.show_status('提示: 标签信息不会被保存')
            self.yolo_predict.save_txt = False
        elif self.save_txt_button.checkState() == Qt.CheckState.Checked:
            self.show_status('提示: 标签信息将会被保存')
            self.yolo_predict.save_txt = True

    def is_show_labels(self):
        if self.show_labels_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.yolo_predict.show_labels = False
        elif self.show_labels_checkbox.checkState() == Qt.CheckState.Checked:
            self.yolo_predict.show_labels = True

    # JSON配置文件初始化
    def load_config(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            save_res = 0
            save_txt = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "save_res": save_res,
                          "save_txt": save_txt
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                save_res = 0
                save_txt = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                save_res = config['save_res']
                save_txt = config['save_txt']
        self.save_res_button.setCheckState(Qt.CheckState(save_res))
        self.yolo_predict.save_res = (False if save_res == 0 else True)
        self.save_txt_button.setCheckState(Qt.CheckState(save_txt))
        self.yolo_predict.save_txt = (False if save_txt == 0 else True)
        self.run_button.setChecked(False)
        self.show_status("欢迎使用车门识别系统")

    # 停止事件（按下停止按钮）
    def stop(self):
        try:
            self.yolo_thread.quit()  # 结束线程
        except:
            pass
        self.yolo_predict.stop_dtc = True
        self.run_button.setChecked(False)  # 恢复按钮状态
        self.save_res_button.setEnabled(True)  # 把保存按钮设置为可用
        self.save_txt_button.setEnabled(True)  # 把保存按钮设置为可用
        self.iou_slider.setEnabled(True)  # 把滑块设置为可用
        self.conf_slider.setEnabled(True)  # 把滑块设置为可用
        self.speed_slider.setEnabled(True)  # 把滑块设置为可用
        self.pre_video.clear()  # 清空视频显示
        self.res_video.clear()  # 清空视频显示
        self.progress_bar.setValue(0)  # 进度条清零
        self.Class_num.setText('--')
        self.Target_num.setText('--')
        self.fps_label.setText('--')
        self.yolo_predict.kinect_selected = 0

    # 检测参数设置
    def change_val(self, x, flag):
        if flag == 'iou_spinbox':
            self.iou_slider.setValue(int(x * 100))
        elif flag == 'iou_slider':
            self.iou_spinbox.setValue(x / 100)
            self.show_status('IOU Threshold: %s' % str(x / 100))
            self.yolo_predict.iou_thres = x / 100
        elif flag == 'conf_spinbox':
            self.conf_slider.setValue(int(x * 100))
        elif flag == 'conf_slider':
            self.conf_spinbox.setValue(x / 100)
            self.show_status('Conf Threshold: %s' % str(x / 100))
            self.yolo_predict.conf_thres = x / 100
        elif flag == 'speed_spinbox':
            self.speed_slider.setValue(x)
        elif flag == 'speed_slider':
            self.speed_spinbox.setValue(x)
            self.show_status('Delay: %s ms' % str(x))
            self.yolo_predict.speed_thres = x  # ms

    # 模型更换
    def change_model(self, x):
        self.select_model = self.model_box.currentText()
        self.yolo_predict.new_model_name = "./weights/%s" % self.select_model
        self.show_status('更改模型：%s' % self.select_model)
        self.Model_name.setText(self.select_model)

    # 循环监测文件夹的文件变化
    def ModelBoxRefre(self):
        pt_list = os.listdir('./weights')
        pt_list = [file for file in pt_list if file.endswith('.pt') or file.endswith('.engine')]
        pt_list.sort(key=lambda x: os.path.getsize('./weights/' + x))
        # 必须排序后再比较，否则列表会一直刷新
        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.model_box.clear()
            self.model_box.addItems(self.pt_list)

    # 获取鼠标位置（用于按住标题栏拖动窗口）
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos

    # 拖动窗口大小时优化调整
    def resizeEvent(self, event):
        # Update Size Grips
        UIFuncitons.resize_grips(self)

    # 退出时退出线程，保存设置
    def closeEvent(self, event):
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.iou_spinbox.value()
        config['conf'] = self.conf_spinbox.value()
        config['rate'] = self.speed_spinbox.value()
        config['save_res'] = (0 if self.save_res_button.checkState() == Qt.Unchecked else 2)
        config['save_txt'] = (0 if self.save_txt_button.checkState() == Qt.Unchecked else 2)
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)

        if self.yolo_thread.isRunning():
            self.yolo_predict.stop_dtc = True
            self.yolo_thread.quit()
            MessageBox(
                self.close_button, title='Note', text='退出中，请等待...', time=2000, auto=True).exec()
            sys.exit(0)
        else:
            sys.exit(0)


if __name__ == "__main__":
    if not os.path.exists('weights'):
        os.mkdir('weights')
    app = QApplication(sys.argv)
    Home = MainWindow()
    Home.show()
    sys.exit(app.exec())
