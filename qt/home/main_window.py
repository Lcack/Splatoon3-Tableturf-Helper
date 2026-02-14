"""
主窗口模块
"""

import os
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
    QGroupBox, QLabel, QStatusBar
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap

from qt.home.controller_widget import ControllerWidget
from qt.home.video_widget import VideoWidget
from qt.home.profile_widget import ProfileWidget
from qt.home.control_widget import ControlWidget
from logger import logger


class MainWindow(QWidget):
    """主控制窗口"""
    
    # 信号定义
    run_started = Signal()
    run_stopped = Signal()
    controller_changed = Signal(object)  # 传递控制器对象
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.controller: Optional[object] = None
        self.capture = None
        self.manager = None
        self.closer = None
        
        self.init_ui()
        self.init_connections()
        
        logger.debug("主窗口初始化完成")
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout()
        
        # 顶部：控制器选择
        controller_group = QGroupBox("控制器设置")
        controller_layout = QVBoxLayout()
        self.controller_widget = ControllerWidget()
        controller_layout.addWidget(self.controller_widget)
        controller_group.setLayout(controller_layout)
        
        main_layout.addWidget(controller_group)
        
        # 中间：视频和配置分割
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：视频显示
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)
        
        # 右侧：配置和控制
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # 配置文件编辑
        profile_group = QGroupBox("配置文件")
        profile_layout = QVBoxLayout()
        self.profile_widget = ProfileWidget()
        profile_layout.addWidget(self.profile_widget)
        profile_group.setLayout(profile_layout)
        
        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout()
        self.control_widget = ControlWidget()
        control_layout.addWidget(self.control_widget)
        control_group.setLayout(control_layout)
        
        right_layout.addWidget(profile_group)
        right_layout.addWidget(control_group)
        right_layout.addStretch()
        
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
        # 设置样式
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
    
    def init_connections(self):
        """初始化信号连接"""
        # 控制器变化
        self.controller_widget.controller_changed.connect(self.on_controller_changed)
        
        # 运行控制
        self.control_widget.run_clicked.connect(self.on_run_clicked)
        self.control_widget.stop_clicked.connect(self.on_stop_clicked)
        
        # 视频源变化
        self.video_widget.source_changed.connect(self.on_video_source_changed)
        
        # 配置文件变化
        self.profile_widget.profile_changed.connect(self.on_profile_changed)
    
    def on_controller_changed(self, controller):
        """控制器变化处理"""
        self.controller = controller
        self.controller_changed.emit(controller)
        
        if controller:
            self.status_label.setText(f"控制器已连接: {type(controller).__name__}")
        else:
            self.status_label.setText("使用虚拟控制器")
        
        logger.debug(f"控制器已更改: {controller}")
    
    def on_run_clicked(self, debug: bool, sleep: bool, stop_at: str):
        """运行按钮点击处理"""
        try:
            # 获取配置文件
            profile_json = self.profile_widget.get_profile_json()
            if not profile_json:
                self.status_label.setText("错误: 配置文件无效")
                return
            
            # 导入必要的模块
            import datetime
            from capture import VideoCapture
            from controller import DummyController
            from tableturf.ai.alpha import Alpha
            from tableturf.manager import TableTurfManager, Profile
            from portal.home.closer import WebCloser
            
            # 设置视频捕获
            source_index = self.video_widget.get_current_source()
            self.capture = VideoCapture(source_index)
            
            # 设置控制器
            if not self.controller:
                self.controller = DummyController()
            
            # 创建 AI
            alpha_ai = Alpha()
            
            # 创建管理器
            self.manager = TableTurfManager(
                self.capture,
                self.controller,
                alpha_ai,
                None,  # 调试器，暂时为 None
            )
            
            # 设置停止时间
            stop_time = None
            if stop_at:
                try:
                    stop_time = datetime.datetime.fromisoformat(stop_at)
                except:
                    pass
            
            # 创建关闭器
            self.closer = WebCloser(stop_time)
            
            # 运行管理器（在新线程中）
            import threading
            def run_manager():
                try:
                    profile = Profile.from_json(profile_json)
                    self.manager.run(profile=profile, closer=self.closer, debug=debug)
                    
                    # 运行完成后处理
                    if sleep and self.controller:
                        self.controller.press_buttons([self.controller.Button.HOME], down=3)
                        self.controller.press_buttons([self.controller.Button.A])
                    
                    self.closer = None
                    self.run_stopped.emit()
                    self.status_label.setText("运行完成")
                    
                except Exception as e:
                    logger.error(f"运行过程中出错: {e}")
                    self.status_label.setText(f"运行错误: {e}")
                    self.run_stopped.emit()
            
            thread = threading.Thread(target=run_manager, daemon=True)
            thread.start()
            
            self.run_started.emit()
            self.status_label.setText("运行中...")
            
        except Exception as e:
            logger.error(f"启动运行失败: {e}")
            self.status_label.setText(f"启动失败: {e}")
    
    def on_stop_clicked(self):
        """停止按钮点击处理"""
        if self.closer:
            self.closer.set_close()
            self.status_label.setText("正在停止...")
        else:
            self.status_label.setText("没有正在运行的任务")
    
    def on_video_source_changed(self, source_index: int):
        """视频源变化处理"""
        self.status_label.setText(f"视频源已切换: {source_index}")
        
        # 这里可以更新视频捕获
        if hasattr(self, 'capture') and self.capture:
            try:
                self.capture.release()
                from capture import VideoCapture
                self.capture = VideoCapture(source_index)
            except Exception as e:
                logger.error(f"切换视频源失败: {e}")
    
    def on_profile_changed(self, profile_text: str):
        """配置文件变化处理"""
        # 可以在这里验证配置文件
        pass
    
    def update_video_frame(self, image):
        """更新视频帧"""
        self.video_widget.update_frame(image)
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.capture:
                self.capture.release()
                self.capture = None
            
            if self.closer:
                self.closer.set_close()
                self.closer = None
            
            logger.debug("主窗口资源已清理")
        except Exception as e:
            logger.error(f"清理资源时出错: {e}")