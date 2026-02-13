import sys
import os
import datetime
import json
from typing import Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTextEdit, QCheckBox, QDateTimeEdit, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QLineEdit, QStatusBar
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal, QTimer
import cv2

from controller import DummyController, Controller, NxbtController, Esp32Controller
import serial.tools.list_ports
from logger import logger
from portal.debug.debugger import web_debugger
from portal.home.closer import WebCloser
from portal.home.keymap import keymap
from qt.home.video_display import VideoDisplay
from qt.profiles.profile_editor import ProfileEditor
from qt.debug import QtDebugger, DebugWindow, qt_debugger
from tableturf.ai.alpha import Alpha
from tableturf.manager import TableTurfManager, Profile





class MainWindow(QMainWindow):
    """
    主窗口
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoTableTurf")
        self.setGeometry(100, 100, 1400, 900)

        # 初始化变量
        self.controller: Optional[Controller] = None
        self.closer: WebCloser = None
        self.debug_window: Optional[DebugWindow] = None

        # 创建主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # 控制器设置区域
        controller_group = QGroupBox("Controller Settings")
        controller_layout = QGridLayout()

        # 控制器类型选择
        controller_layout.addWidget(QLabel("Controller Type:"), 0, 0)
        self.controller_type_combo = QComboBox()
        self.controller_type_combo.addItems(["nxbt", "esp32"])
        self.controller_type_combo.currentIndexChanged.connect(self.on_controller_type_changed)
        controller_layout.addWidget(self.controller_type_combo, 0, 1)

        # Nxbt配置
        self.nxbt_group = QGroupBox("Nxbt Configuration")
        nxbt_layout = QVBoxLayout()
        nxbt_layout.addWidget(QLabel("Controller Endpoint:"))
        self.controller_endpoint_edit = QLineEdit("http://192.168.50.101:5000/")
        nxbt_layout.addWidget(self.controller_endpoint_edit)
        self.nxbt_group.setLayout(nxbt_layout)
        controller_layout.addWidget(self.nxbt_group, 1, 0, 1, 2)

        # ESP32配置
        self.esp32_group = QGroupBox("ESP32 Configuration")
        esp32_layout = QVBoxLayout()
        esp32_layout.addWidget(QLabel("Serial Port:"))
        self.port_combo = QComboBox()
        self.update_serial_ports()
        esp32_layout.addWidget(self.port_combo)
        self.flash_button = QPushButton("Flash Firmware")
        self.flash_button.clicked.connect(self.on_flash_clicked)
        esp32_layout.addWidget(self.flash_button)
        self.esp32_group.setLayout(esp32_layout)
        self.esp32_group.hide()
        controller_layout.addWidget(self.esp32_group, 1, 0, 1, 2)

        # 连接按钮
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.on_connect_clicked)
        controller_layout.addWidget(self.connect_button, 2, 0, 1, 2)

        controller_group.setLayout(controller_layout)
        main_layout.addWidget(controller_group)

        # 视频显示组件
        self.video_display = VideoDisplay()
        self.video_display.source_combo.currentIndexChanged.connect(self.on_source_changed)
        main_layout.addWidget(self.video_display)
        
        # 配置编辑器
        self.profile_editor = ProfileEditor()
        main_layout.addWidget(self.profile_editor)

        # 运行控制区域
        control_layout = QVBoxLayout()

        # 停止时间设置
        stop_layout = QHBoxLayout()
        stop_layout.addWidget(QLabel("Stop at:"))
        self.stop_at_edit = QDateTimeEdit()
        self.stop_at_edit.setDateTime(datetime.datetime.now())
        stop_layout.addWidget(self.stop_at_edit)
        stop_layout.addStretch()
        control_layout.addLayout(stop_layout)

        # 按钮区域
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.on_stop_clicked)
        button_layout.addWidget(self.stop_button)

        self.sleep_checkbox = QCheckBox("Turn off Switch after stop")
        button_layout.addWidget(self.sleep_checkbox)

        control_layout.addLayout(button_layout)

        # 调试选项
        debug_layout = QHBoxLayout()
        self.debug_checkbox = QCheckBox("Debug")
        debug_layout.addWidget(self.debug_checkbox)
        
        # 调试窗口按钮
        self.debug_window_button = QPushButton("Open Debug Window")
        self.debug_window_button.clicked.connect(self.on_open_debug_window)
        debug_layout.addWidget(self.debug_window_button)
        
        debug_layout.addStretch()
        control_layout.addLayout(debug_layout)

        main_layout.addLayout(control_layout)

        # 按键映射表格
        keymap_group = QGroupBox("Keymap")
        keymap_layout = QVBoxLayout()
        self.keymap_table = QTableWidget()
        self.keymap_table.setColumnCount(2)
        self.keymap_table.setHorizontalHeaderLabels(["Keyboard", "Controller"])
        self.keymap_table.setRowCount(len(keymap))
        for row, (key, value) in enumerate(keymap.items()):
            self.keymap_table.setItem(row, 0, QTableWidgetItem(key))
            self.keymap_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.keymap_table.resizeColumnsToContents()
        keymap_layout.addWidget(self.keymap_table)
        keymap_group.setLayout(keymap_layout)
        main_layout.addWidget(keymap_group)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 设置中心组件
        self.setCentralWidget(central_widget)

        # 开始视频捕获
        self.video_display.start_video_capture()



    def update_serial_ports(self):
        """
        更新可用的串口端口
        """
        self.port_combo.clear()
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(port.device)
        self.port_combo.addItems(ports)

    def on_controller_type_changed(self, index):
        """
        控制器类型改变时的处理
        """
        if self.controller_type_combo.currentText() == "nxbt":
            self.nxbt_group.show()
            self.esp32_group.hide()
        else:
            self.nxbt_group.hide()
            self.esp32_group.show()

    def on_source_changed(self, index):
        """
        视频源改变时的处理
        """
        source = self.video_display.source_combo.currentData()
        self.video_display.update_capture_source(source)

    def on_connect_clicked(self):
        """
        连接控制器
        """
        controller_type = self.controller_type_combo.currentText()
        endpoint = self.controller_endpoint_edit.text()
        port = self.port_combo.currentText()

        if controller_type == 'esp32' and port:
            self.controller = Esp32Controller(port)
            self.status_bar.showMessage("ESP32 controller connected")
        elif endpoint != '':
            self.controller = NxbtController(endpoint)
            self.status_bar.showMessage("Nxbt controller connected")
        else:
            self.controller = DummyController()
            self.status_bar.showMessage("Dummy controller connected")

    def on_flash_clicked(self):
        """
        烧录ESP32固件
        """
        from controller.esp32_flasher import flash_esp32_firmware

        self.status_bar.showMessage('Flashing ESP32 firmware...')
        success, message = flash_esp32_firmware()
        self.status_bar.showMessage(message)

    def on_run_clicked(self):
        """
        运行游戏
        """
        debug = self.debug_checkbox.isChecked()
        sleep = self.sleep_checkbox.isChecked()
        profile = self.profile_editor.get_profile_text()
        stop_at = self.stop_at_edit.dateTime().toString('yyyy-MM-ddTHH:mm:ss')

        try:
            time = datetime.datetime.fromisoformat(stop_at)
        except:
            time = None

        logger.debug(f'Run: profile={profile}, sleep={sleep}, debug={debug}, stop_at={stop_at}')

        alpha_ai = Alpha()
        manager = TableTurfManager(
            self.video_display.capture,
            self.controller if self.controller is not None else DummyController(),
            alpha_ai,
            qt_debugger,
        )

        if self.closer is None:
            self.closer = WebCloser(time)

        manager.run(profile=Profile.from_json(profile), closer=self.closer, debug=debug)
        self.closer = None

        if sleep and self.controller:
            self.controller.press_buttons([Controller.Button.HOME], down=3)
            self.controller.press_buttons([Controller.Button.A])

    def on_stop_clicked(self):
        """
        停止运行
        """
        if self.closer is not None:
            self.closer.set_close()
            self.status_bar.showMessage('Stopping...')



    def on_open_debug_window(self):
        """
        打开调试窗口
        """
        if not self.debug_window:
            self.debug_window = DebugWindow()
        self.debug_window.show()

    def closeEvent(self, event):
        """
        窗口关闭事件
        """
        self.video_display.stop_video_capture()
        # 关闭调试窗口
        if self.debug_window:
            self.debug_window.close()
        event.accept()