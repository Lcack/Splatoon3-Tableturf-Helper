"""
控制部件 - 运行、停止和其他控制功能
"""

import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QCheckBox, QDateTimeEdit, QLabel,
    QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
from PySide6.QtGui import QFont

from logger import logger


class ControlWidget(QWidget):
    """控制部件"""
    
    # 信号定义
    run_clicked = Signal(bool, bool, str)  # debug, sleep, stop_at
    stop_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.is_running = False
        self.start_time: Optional[datetime.datetime] = None
        self.update_timer = QTimer()
        
        self.init_ui()
        self.init_connections()
        
        logger.debug("控制部件初始化完成")
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 运行时间设置
        time_group = QGroupBox("运行时间设置")
        time_layout = QVBoxLayout()
        
        # 停止时间
        stop_layout = QHBoxLayout()
        stop_layout.addWidget(QLabel("停止时间:"))
        
        self.stop_datetime = QDateTimeEdit()
        self.stop_datetime.setDateTime(QDateTime.currentDateTime().addSecs(3600))  # 默认1小时后
        self.stop_datetime.setCalendarPopup(True)
        self.stop_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        stop_layout.addWidget(self.stop_datetime)
        
        time_layout.addLayout(stop_layout)
        
        # 立即停止选项
        self.no_stop_checkbox = QCheckBox("不自动停止")
        self.no_stop_checkbox.stateChanged.connect(self.on_no_stop_changed)
        time_layout.addWidget(self.no_stop_checkbox)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # 选项
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout()
        
        self.debug_checkbox = QCheckBox("调试模式")
        options_layout.addWidget(self.debug_checkbox)
        
        self.sleep_checkbox = QCheckBox("运行后休眠 Switch")
        options_layout.addWidget(self.sleep_checkbox)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("运行")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        button_layout.addWidget(self.run_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # 运行状态
        status_group = QGroupBox("运行状态")
        status_layout = QVBoxLayout()
        
        # 运行时间
        self.runtime_label = QLabel("运行时间: 00:00:00")
        self.runtime_label.setFont(QFont("Monospace", 10))
        status_layout.addWidget(self.runtime_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.progress_bar)
        
        # 状态信息
        self.status_info_label = QLabel("就绪")
        self.status_info_label.setWordWrap(True)
        status_layout.addWidget(self.status_info_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        self.setLayout(layout)
    
    def init_connections(self):
        """初始化信号连接"""
        self.run_button.clicked.connect(self.on_run_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        
        # 更新定时器
        self.update_timer.timeout.connect(self.update_runtime)
    
    def on_no_stop_changed(self, state):
        """不自动停止选项变化处理"""
        self.stop_datetime.setEnabled(state == Qt.Unchecked)
    
    def on_run_clicked(self):
        """运行按钮点击处理"""
        if self.is_running:
            QMessageBox.warning(self, "警告", "已有任务正在运行")
            return
        
        # 获取参数
        debug = self.debug_checkbox.isChecked()
        sleep = self.sleep_checkbox.isChecked()
        
        # 获取停止时间
        stop_at = ""
        if not self.no_stop_checkbox.isChecked():
            stop_datetime = self.stop_datetime.dateTime()
            stop_at = stop_datetime.toString("yyyy-MM-ddTHH:mm:ss")
        
        # 验证停止时间
        if stop_at:
            try:
                stop_time = datetime.datetime.fromisoformat(stop_at)
                if stop_time <= datetime.datetime.now():
                    QMessageBox.warning(self, "警告", "停止时间必须晚于当前时间")
                    return
            except:
                QMessageBox.warning(self, "警告", "停止时间格式无效")
                return
        
        # 确认对话框
        confirm_msg = "确认开始运行？"
        if stop_at:
            confirm_msg += f"\n停止时间: {stop_at}"
        if sleep:
            confirm_msg += "\n运行后将休眠 Switch"
        
        reply = QMessageBox.question(
            self, '确认运行',
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 更新状态
            self.is_running = True
            self.start_time = datetime.datetime.now()
            
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            self.status_info_label.setText("正在启动...")
            self.status_info_label.setStyleSheet("color: orange;")
            
            # 启动定时器
            self.update_timer.start(1000)  # 每秒更新一次
            
            # 发送信号
            self.run_clicked.emit(debug, sleep, stop_at)
            
            logger.info(f"开始运行: debug={debug}, sleep={sleep}, stop_at={stop_at}")
    
    def on_stop_clicked(self):
        """停止按钮点击处理"""
        if not self.is_running:
            return
        
        reply = QMessageBox.question(
            self, '确认停止',
            '确定要停止当前运行的任务吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 发送停止信号
            self.stop_clicked.emit()
            
            self.status_info_label.setText("正在停止...")
            self.status_info_label.setStyleSheet("color: orange;")
            
            logger.info("用户请求停止运行")
    
    def update_runtime(self):
        """更新运行时间"""
        if self.start_time and self.is_running:
            elapsed = datetime.datetime.now() - self.start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            runtime_str = f"运行时间: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            self.runtime_label.setText(runtime_str)
            
            # 更新进度条（如果有停止时间）
            if not self.no_stop_checkbox.isChecked():
                stop_datetime = self.stop_datetime.dateTime()
                stop_time = stop_datetime.toPyDateTime()
                
                total_seconds = (stop_time - self.start_time).total_seconds()
                if total_seconds > 0:
                    elapsed_seconds = elapsed.total_seconds()
                    progress = min(100, int((elapsed_seconds / total_seconds) * 100))
                    self.progress_bar.setValue(progress)
    
    def set_running_state(self, is_running: bool):
        """设置运行状态"""
        self.is_running = is_running
        
        if is_running:
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.start_time = datetime.datetime.now()
            self.update_timer.start(1000)
            
            self.status_info_label.setText("运行中...")
            self.status_info_label.setStyleSheet("color: green;")
        else:
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.update_timer.stop()
            
            self.status_info_label.setText("已停止")
            self.status_info_label.setStyleSheet("color: red;")
            
            # 重置运行时间
            self.runtime_label.setText("运行时间: 00:00:00")
            self.progress_bar.setValue(0)
    
    def set_status_message(self, message: str, is_error: bool = False):
        """设置状态消息"""
        self.status_info_label.setText(message)
        if is_error:
            self.status_info_label.setStyleSheet("color: red;")
        else:
            self.status_info_label.setStyleSheet("color: black;")
    
    def set_progress(self, value: int):
        """设置进度条值"""
        self.progress_bar.setValue(value)
    
    def cleanup(self):
        """清理资源"""
        self.update_timer.stop()
        logger.debug("控制部件资源已清理")