"""
视频部件 - 显示视频流和选择视频源
"""

import cv2
import numpy as np
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QLabel, QPushButton
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap

from logger import logger


class VideoWidget(QWidget):
    """视频显示部件"""
    
    # 信号定义
    source_changed = Signal(int)  # 传递视频源索引
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_source = 0
        self.capture = None
        self.video_timer = QTimer()
        
        self.init_ui()
        self.init_connections()
        
        # 刷新视频源列表
        self.refresh_sources()
        
        logger.debug("视频部件初始化完成")
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 视频源选择
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("视频源:"))
        
        self.source_combo = QComboBox()
        source_layout.addWidget(self.source_combo)
        
        self.refresh_btn = QPushButton("刷新")
        source_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(source_layout)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
        """)
        
        # 设置默认文本
        self.video_label.setText("无视频信号")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 2px solid #cccccc;
                border-radius: 5px;
                color: white;
                font-size: 16px;
            }
        """)
        
        layout.addWidget(self.video_label)
        
        self.setLayout(layout)
    
    def init_connections(self):
        """初始化信号连接"""
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        self.refresh_btn.clicked.connect(self.refresh_sources)
        
        # 视频定时器
        self.video_timer.timeout.connect(self.update_video_frame)
    
    def refresh_sources(self):
        """刷新视频源列表"""
        try:
            sources = self.list_available_sources()
            self.source_combo.clear()
            
            if sources:
                for source in sources:
                    self.source_combo.addItem(f"摄像头 {source}", source)
                logger.debug(f"找到视频源: {sources}")
            else:
                self.source_combo.addItem("未找到视频源", -1)
                logger.warning("未找到视频源")
                
        except Exception as e:
            logger.error(f"刷新视频源失败: {e}")
    
    def list_available_sources(self) -> List[int]:
        """列出可用的视频源"""
        sources = []
        index = 0
        
        # 尝试打开前10个视频源
        while index < 10:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    sources.append(index)
                cap.release()
            index += 1
        
        return sources
    
    def on_source_changed(self, index: int):
        """视频源变化处理"""
        if index < 0:
            return
        
        source_data = self.source_combo.itemData(index)
        if source_data == -1:
            return
        
        new_source = source_data
        if new_source != self.current_source:
            self.current_source = new_source
            
            # 停止当前视频
            self.stop_video()
            
            # 启动新视频
            self.start_video(new_source)
            
            # 发送信号
            self.source_changed.emit(new_source)
            
            logger.debug(f"视频源已切换: {new_source}")
    
    def start_video(self, source_index: int):
        """启动视频捕获"""
        try:
            self.capture = cv2.VideoCapture(source_index)
            
            if not self.capture.isOpened():
                logger.error(f"无法打开视频源: {source_index}")
                self.video_label.setText(f"无法打开摄像头 {source_index}")
                return
            
            # 设置视频参数
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 启动定时器
            self.video_timer.start(30)  # 约30fps
            
            logger.info(f"视频捕获已启动: 源 {source_index}")
            
        except Exception as e:
            logger.error(f"启动视频捕获失败: {e}")
            self.video_label.setText(f"启动失败: {e}")
    
    def stop_video(self):
        """停止视频捕获"""
        if self.video_timer.isActive():
            self.video_timer.stop()
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.debug("视频捕获已停止")
    
    def update_video_frame(self):
        """更新视频帧"""
        if not self.capture or not self.capture.isOpened():
            return
        
        try:
            ret, frame = self.capture.read()
            if ret:
                # 转换颜色空间 BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 调整大小以适应显示区域
                h, w, ch = frame_rgb.shape
                target_width = self.video_label.width()
                target_height = self.video_label.height()
                
                # 保持宽高比
                if w > target_width or h > target_height:
                    scale = min(target_width / w, target_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                    h, w, ch = frame_rgb.shape
                
                # 创建 QImage
                bytes_per_line = ch * w
                qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # 创建 QPixmap 并显示
                pixmap = QPixmap.fromImage(qimage)
                self.video_label.setPixmap(pixmap)
                
        except Exception as e:
            logger.error(f"更新视频帧失败: {e}")
            self.video_label.setText("视频帧错误")
    
    def get_current_source(self) -> int:
        """获取当前视频源"""
        return self.current_source
    
    def update_frame(self, image):
        """从外部更新帧（用于主窗口调用）"""
        if image is None:
            return
        
        try:
            # 确保图像是 RGB 格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 假设是 BGR，转换为 RGB
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = image
            
            h, w, ch = frame_rgb.shape
            target_width = self.video_label.width()
            target_height = self.video_label.height()
            
            # 保持宽高比
            if w > target_width or h > target_height:
                scale = min(target_width / w, target_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                h, w, ch = frame_rgb.shape
            
            # 创建 QImage
            bytes_per_line = ch * w
            qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 创建 QPixmap 并显示
            pixmap = QPixmap.fromImage(qimage)
            self.video_label.setPixmap(pixmap)
            
        except Exception as e:
            logger.error(f"更新外部帧失败: {e}")
    
    def resizeEvent(self, event):
        """窗口大小变化事件"""
        super().resizeEvent(event)
        # 可以在这里调整视频显示
    
    def cleanup(self):
        """清理资源"""
        self.stop_video()
        logger.debug("视频部件资源已清理")