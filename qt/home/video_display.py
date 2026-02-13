import sys
import os
from typing import Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal
import cv2

from capture import VideoCapture
from portal.home.capture import ThreadSafeCapture
from logger import logger


class VideoCaptureThread(QThread):
    """
    视频捕获线程
    """
    image_ready = Signal(QImage)

    def __init__(self, capture):
        super().__init__()
        self.capture = capture
        self.running = True

    def run(self):
        while self.running:
            try:
                img = self.capture.capture()
                if img is not None:
                    # 转换为QImage
                    height, width, channel = img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.image_ready.emit(q_img)
            except Exception as e:
                logger.error(f"Video capture error: {e}")
            # 短暂休眠，避免CPU占用过高
            self.msleep(30)

    def stop(self):
        self.running = False
        self.wait()


class VideoDisplay(QWidget):
    """
    视频显示组件
    """
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 视频源选择
        self.source_layout = QHBoxLayout()
        self.source_layout.addWidget(QLabel("Screen Source:"))
        self.source_combo = QComboBox()
        self.available_sources = self.list_available_source()
        for source in self.available_sources:
            self.source_combo.addItem(str(source), source)
        self.source_layout.addWidget(self.source_combo)
        self.source_layout.addStretch()
        self.layout.addLayout(self.source_layout)
        
        # 视频显示
        self.display_layout = QHBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(1280, 720)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.display_layout.addWidget(self.video_label)
        self.layout.addLayout(self.display_layout)
        
        # 初始化变量
        self.capture = ThreadSafeCapture(VideoCapture(0))
        self.video_thread = None
    
    def list_available_source(self):
        """
        列出所有可用的视频源
        """
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr
    
    def start_video_capture(self):
        """
        开始视频捕获
        """
        if self.video_thread:
            self.video_thread.stop()

        self.video_thread = VideoCaptureThread(self.capture)
        self.video_thread.image_ready.connect(self.update_video_display)
        self.video_thread.start()
    
    def update_video_display(self, q_img):
        """
        更新视频显示
        """
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
    
    def update_capture_source(self, source):
        """
        更新视频捕获源
        """
        self.capture.update_capture(VideoCapture(source))
        logger.debug(f'Source changed: {source}')
    
    def stop_video_capture(self):
        """
        停止视频捕获
        """
        if self.video_thread:
            self.video_thread.stop()