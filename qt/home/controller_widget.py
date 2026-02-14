"""
控制器部件 - 处理控制器选择（虚拟/NXBT/ESP32）
"""

import os
from typing import Optional
from enum import Enum

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QRadioButton, QButtonGroup, QLineEdit, QPushButton,
    QComboBox, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer

from logger import logger


class ControllerType(Enum):
    """控制器类型枚举"""
    DUMMY = "dummy"
    NXBT = "nxbt"
    ESP32 = "esp32"


class ControllerWidget(QWidget):
    """控制器选择部件"""
    
    # 信号定义
    controller_changed = Signal(object)  # 传递控制器对象
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.controller: Optional[object] = None
        self.controller_type: ControllerType = ControllerType.DUMMY
        
        self.init_ui()
        self.init_connections()
        
        # 定时器用于更新 ESP32 状态
        self.esp32_status_timer = QTimer()
        self.esp32_status_timer.timeout.connect(self.update_esp32_status)
        
        logger.debug("控制器部件初始化完成")
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 控制器类型选择
        type_group = QGroupBox("控制器类型")
        type_layout = QVBoxLayout()
        
        self.button_group = QButtonGroup(self)
        
        # 虚拟控制器
        self.dummy_radio = QRadioButton("虚拟控制器")
        self.dummy_radio.setChecked(True)
        self.button_group.addButton(self.dummy_radio)
        type_layout.addWidget(self.dummy_radio)
        
        # NXBT 控制器
        self.nxbt_radio = QRadioButton("NXBT (蓝牙)")
        self.button_group.addButton(self.nxbt_radio)
        
        nxbt_layout = QHBoxLayout()
        nxbt_layout.addWidget(self.nxbt_radio)
        
        self.nxbt_endpoint = QLineEdit("http://192.168.50.101:5000/")
        self.nxbt_endpoint.setPlaceholderText("NXBT 服务器地址")
        self.nxbt_endpoint.setEnabled(False)
        nxbt_layout.addWidget(self.nxbt_endpoint)
        
        self.nxbt_connect_btn = QPushButton("连接")
        self.nxbt_connect_btn.setEnabled(False)
        nxbt_layout.addWidget(self.nxbt_connect_btn)
        
        type_layout.addLayout(nxbt_layout)
        
        # ESP32 控制器
        self.esp32_radio = QRadioButton("ESP32 (串口)")
        self.button_group.addButton(self.esp32_radio)
        
        esp32_layout = QVBoxLayout()
        esp32_layout.addWidget(self.esp32_radio)
        
        # 串口选择
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("串口:"))
        
        self.esp32_port_combo = QComboBox()
        self.esp32_port_combo.setEnabled(False)
        port_layout.addWidget(self.esp32_port_combo)
        
        self.esp32_refresh_btn = QPushButton("刷新")
        self.esp32_refresh_btn.setEnabled(False)
        port_layout.addWidget(self.esp32_refresh_btn)
        
        esp32_layout.addLayout(port_layout)
        
        # 连接控制
        connect_layout = QHBoxLayout()
        self.esp32_connect_btn = QPushButton("连接")
        self.esp32_connect_btn.setEnabled(False)
        connect_layout.addWidget(self.esp32_connect_btn)
        
        self.esp32_disconnect_btn = QPushButton("断开")
        self.esp32_disconnect_btn.setEnabled(False)
        connect_layout.addWidget(self.esp32_disconnect_btn)
        
        # 状态显示
        self.esp32_status_label = QLabel("未连接")
        self.esp32_status_label.setStyleSheet("color: red; font-weight: bold;")
        connect_layout.addWidget(self.esp32_status_label)
        
        esp32_layout.addLayout(connect_layout)
        
        type_layout.addLayout(esp32_layout)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        self.setLayout(layout)
    
    def init_connections(self):
        """初始化信号连接"""
        # 单选按钮变化
        self.dummy_radio.toggled.connect(lambda checked: self.on_controller_type_changed(ControllerType.DUMMY, checked))
        self.nxbt_radio.toggled.connect(lambda checked: self.on_controller_type_changed(ControllerType.NXBT, checked))
        self.esp32_radio.toggled.connect(lambda checked: self.on_controller_type_changed(ControllerType.ESP32, checked))
        
        # NXBT 连接按钮
        self.nxbt_connect_btn.clicked.connect(self.connect_nxbt)
        
        # ESP32 按钮
        self.esp32_refresh_btn.clicked.connect(self.refresh_esp32_ports)
        self.esp32_connect_btn.clicked.connect(self.connect_esp32)
        self.esp32_disconnect_btn.clicked.connect(self.disconnect_esp32)
    
    def on_controller_type_changed(self, controller_type: ControllerType, checked: bool):
        """控制器类型变化处理"""
        if not checked:
            return
        
        self.controller_type = controller_type
        
        # 更新控件状态
        self.nxbt_endpoint.setEnabled(controller_type == ControllerType.NXBT)
        self.nxbt_connect_btn.setEnabled(controller_type == ControllerType.NXBT)
        
        esp32_enabled = controller_type == ControllerType.ESP32
        self.esp32_port_combo.setEnabled(esp32_enabled)
        self.esp32_refresh_btn.setEnabled(esp32_enabled)
        self.esp32_connect_btn.setEnabled(esp32_enabled)
        
        # 停止 ESP32 状态定时器
        if controller_type != ControllerType.ESP32:
            self.esp32_status_timer.stop()
        
        # 创建控制器
        self.create_controller()
        
        logger.debug(f"控制器类型已更改为: {controller_type}")
    
    def create_controller(self):
        """创建控制器对象"""
        try:
            # 清理现有控制器
            if self.controller:
                try:
                    if hasattr(self.controller, 'disconnect'):
                        self.controller.disconnect()
                except:
                    pass
                self.controller = None
            
            # 创建新控制器
            if self.controller_type == ControllerType.DUMMY:
                from controller import DummyController
                self.controller = DummyController()
                
            elif self.controller_type == ControllerType.NXBT:
                # NXBT 控制器需要连接后才创建
                self.controller = None
                return
                
            elif self.controller_type == ControllerType.ESP32:
                # ESP32 控制器需要连接后才创建
                self.controller = None
                return
            
            # 发送信号
            self.controller_changed.emit(self.controller)
            logger.debug(f"控制器已创建: {type(self.controller).__name__}")
            
        except Exception as e:
            logger.error(f"创建控制器失败: {e}")
            QMessageBox.warning(self, "错误", f"创建控制器失败: {e}")
    
    def connect_nxbt(self):
        """连接 NXBT"""
        endpoint = self.nxbt_endpoint.text().strip()
        if not endpoint:
            QMessageBox.warning(self, "警告", "请输入 NXBT 服务器地址")
            return
        
        try:
            from controller import NxbtController
            self.controller = NxbtController(endpoint)
            
            # 测试连接
            # 这里可以添加一个简单的测试，比如发送一个测试命令
            
            self.controller_changed.emit(self.controller)
            self.nxbt_connect_btn.setText("已连接")
            self.nxbt_connect_btn.setEnabled(False)
            
            logger.info(f"NXBT 连接成功: {endpoint}")
            
        except Exception as e:
            logger.error(f"连接 NXBT 失败: {e}")
            QMessageBox.critical(self, "连接失败", f"连接 NXBT 失败: {e}")
    
    def refresh_esp32_ports(self):
        """刷新 ESP32 串口列表"""
        try:
            from controller.esp32 import ESP32Controller
            
            # 创建临时控制器获取串口列表
            temp_controller = ESP32Controller(auto_connect=False)
            ports = temp_controller.list_available_ports()
            
            self.esp32_port_combo.clear()
            if ports:
                for port in ports:
                    self.esp32_port_combo.addItem(port)
                logger.debug(f"找到串口: {ports}")
            else:
                self.esp32_port_combo.addItem("未找到可用串口")
                logger.warning("未找到可用串口")
                
        except Exception as e:
            logger.error(f"刷新串口失败: {e}")
            QMessageBox.warning(self, "错误", f"刷新串口失败: {e}")
    
    def connect_esp32(self):
        """连接 ESP32"""
        port = self.esp32_port_combo.currentText()
        if not port or port == "未找到可用串口":
            QMessageBox.warning(self, "警告", "请选择有效的串口")
            return
        
        try:
            from controller.esp32 import ESP32Controller
            
            self.controller = ESP32Controller(port_name=port, auto_connect=True)
            
            if self.controller.is_connected():
                self.controller_changed.emit(self.controller)
                
                self.esp32_connect_btn.setEnabled(False)
                self.esp32_disconnect_btn.setEnabled(True)
                self.esp32_status_label.setText("已连接")
                self.esp32_status_label.setStyleSheet("color: green; font-weight: bold;")
                
                # 启动状态定时器
                self.esp32_status_timer.start(2000)  # 每2秒更新一次
                
                logger.info(f"ESP32 连接成功: {port}")
            else:
                self.controller = None
                QMessageBox.warning(self, "连接失败", "ESP32 连接失败，请检查设备和串口")
                logger.warning(f"ESP32 连接失败: {port}")
                
        except Exception as e:
            logger.error(f"连接 ESP32 失败: {e}")
            QMessageBox.critical(self, "连接失败", f"连接 ESP32 失败: {e}")
    
    def disconnect_esp32(self):
        """断开 ESP32 连接"""
        try:
            if self.controller and hasattr(self.controller, 'disconnect'):
                self.controller.disconnect()
            
            self.controller = None
            self.controller_changed.emit(None)
            
            self.esp32_connect_btn.setEnabled(True)
            self.esp32_disconnect_btn.setEnabled(False)
            self.esp32_status_label.setText("未连接")
            self.esp32_status_label.setStyleSheet("color: red; font-weight: bold;")
            
            # 停止状态定时器
            self.esp32_status_timer.stop()
            
            logger.info("ESP32 已断开连接")
            
        except Exception as e:
            logger.error(f"断开 ESP32 连接失败: {e}")
    
    def update_esp32_status(self):
        """更新 ESP32 状态"""
        if self.controller and hasattr(self.controller, 'is_connected'):
            connected = self.controller.is_connected()
            
            if connected:
                self.esp32_status_label.setText("已连接")
                self.esp32_status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.esp32_status_label.setText("连接断开")
                self.esp32_status_label.setStyleSheet("color: orange; font-weight: bold;")
    
    def get_controller(self):
        """获取当前控制器"""
        return self.controller
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.controller and hasattr(self.controller, 'disconnect'):
                self.controller.disconnect()
            
            self.esp32_status_timer.stop()
            
            logger.debug("控制器部件资源已清理")
        except Exception as e:
            logger.error(f"清理控制器部件资源时出错: {e}")