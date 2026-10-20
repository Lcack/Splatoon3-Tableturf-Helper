"""
ESP32 控制器主模块
实现 Controller 接口，使用 ESP32 串口通信
"""

import time
import threading
from typing import List, Optional
from enum import Enum

from controller import Controller
from controller.esp32_serial import ESP32SerialController, SerialStatus
from controller.esp32_input import ESP32InputEmulator
from logger import logger


class ESP32Controller(Controller):
    """
    ESP32 控制器
    通过串口与 ESP32 通信，模拟 Switch 手柄输入
    """
    
    # 按钮映射：Controller.Button -> ESP32InputEmulator.Button
    _BUTTON_MAP = {
        Controller.Button.Y: ESP32InputEmulator.Button.Y,
        Controller.Button.X: ESP32InputEmulator.Button.X,
        Controller.Button.B: ESP32InputEmulator.Button.B,
        Controller.Button.A: ESP32InputEmulator.Button.A,
        Controller.Button.JCL_SR: ESP32InputEmulator.Button.SR,
        Controller.Button.JCL_SL: ESP32InputEmulator.Button.SL,
        Controller.Button.R: ESP32InputEmulator.Button.R,
        Controller.Button.ZR: ESP32InputEmulator.Button.ZR,
        Controller.Button.MINUS: ESP32InputEmulator.Button.MINUS,
        Controller.Button.PLUS: ESP32InputEmulator.Button.PLUS,
        Controller.Button.R_STICK_PRESS: ESP32InputEmulator.Button.R_CLICK,
        Controller.Button.L_STICK_PRESS: ESP32InputEmulator.Button.L_CLICK,
        Controller.Button.HOME: ESP32InputEmulator.Button.HOME,
        Controller.Button.CAPTURE: ESP32InputEmulator.Button.CAPTURE,
        Controller.Button.DPAD_DOWN: ESP32InputEmulator.DPad.DOWN,
        Controller.Button.DPAD_UP: ESP32InputEmulator.DPad.UP,
        Controller.Button.DPAD_RIGHT: ESP32InputEmulator.DPad.RIGHT,
        Controller.Button.DPAD_LEFT: ESP32InputEmulator.DPad.LEFT,
        Controller.Button.JCR_SR: ESP32InputEmulator.Button.SR,
        Controller.Button.JCR_SL: ESP32InputEmulator.Button.SL,
        Controller.Button.L: ESP32InputEmulator.Button.L,
        Controller.Button.ZL: ESP32InputEmulator.Button.ZL,
    }
    
    # 方向键映射
    _DPAD_MAP = {
        Controller.Button.DPAD_UP: ESP32InputEmulator.DPad.UP,
        Controller.Button.DPAD_DOWN: ESP32InputEmulator.DPad.DOWN,
        Controller.Button.DPAD_LEFT: ESP32InputEmulator.DPad.LEFT,
        Controller.Button.DPAD_RIGHT: ESP32InputEmulator.DPad.RIGHT,
    }
    
    def __init__(self, port_name: Optional[str] = None, auto_connect: bool = True):
        """
        初始化 ESP32 控制器
        
        Args:
            port_name: 串口名称（如 '/dev/ttyUSB0' 或 'COM3'）
            auto_connect: 是否自动连接
        """
        self._serial = ESP32SerialController()
        self._port_name = port_name
        self._connected = False
        
        if auto_connect and port_name:
            self.connect(port_name)
    
    def connect(self, port_name: Optional[str] = None) -> bool:
        """
        连接到 ESP32
        
        Args:
            port_name: 串口名称，如果为 None 则使用初始化时的端口
        
        Returns:
            连接是否成功
        """
        if port_name:
            self._port_name = port_name
        
        if not self._port_name:
            logger.error("未指定串口名称")
            return False
        
        logger.info(f"尝试连接到 ESP32: {self._port_name}")
        
        # 打开并同步
        success = self._serial.open_and_sync(self._port_name)
        
        if success:
            self._connected = True
            logger.info("ESP32 连接成功")
        else:
            self._connected = False
            logger.error("ESP32 连接失败")
        
        return success
    
    def disconnect(self):
        """断开连接"""
        self._serial.close()
        self._connected = False
        logger.info("ESP32 已断开连接")
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected and self._serial.is_connected() and self._serial.is_synced()
    
    def list_available_ports(self) -> List[str]:
        """列出可用的串口"""
        return self._serial.list_ports()
    
    def press_buttons(self, buttons: List[Controller.Button], 
                     down: float = 0.08, up: float = 0.08, block: bool = True) -> bool:
        """
        按下按钮
        
        Args:
            buttons: 要按下的按钮列表
            down: 按下持续时间（秒）
            up: 释放后等待时间（秒）
            block: 是否阻塞
        
        Returns:
            操作是否成功
        """
        logger.debug(f"press_buttons: buttons={buttons}, down={down}, up={up}, block={block}")
        
        if not self.is_connected():
            logger.error("ESP32 未连接")
            return False
        
        # 分离方向键和普通按钮
        dpad_buttons = []
        normal_buttons = []
        
        for button in buttons:
            if button in self._DPAD_MAP:
                dpad_buttons.append(button)
            else:
                normal_buttons.append(button)
        
        # 转换按钮
        esp32_buttons = []
        for button in normal_buttons:
            if button in self._BUTTON_MAP:
                esp32_buttons.append(self._BUTTON_MAP[button])
        
        # 处理方向键（只取第一个方向键，因为不能同时按多个方向）
        dpad = ESP32InputEmulator.DPad.CENTER
        if dpad_buttons:
            # 如果有多个方向键，组合它们（如 UP+LEFT）
            dpad_value = 0
            for button in dpad_buttons:
                dpad_value |= self._DPAD_MAP[button].value
            dpad = ESP32InputEmulator.DPad(dpad_value)
        
        def _press():
            # 按下按钮
            success = self._serial.send_buttons(
                buttons=esp32_buttons,
                dpad=dpad,
                temporary=False
            )
            
            if success and down > 0:
                time.sleep(down)
            
            # 释放按钮（发送空白命令）
            if success:
                success = self._serial.send_command(ESP32InputEmulator.NO_INPUT)
            
            if success and up > 0:
                time.sleep(up)
            
            return success
        
        if block:
            return _press()
        else:
            thread = threading.Thread(target=_press)
            thread.start()
            return True
    
    def tilt_stick(self, stick: Controller.Stick, x: int, y: int, 
                  tilted: float = 0.1, released: float = 0.1, block: bool = True) -> bool:
        """
        倾斜摇杆
        
        Args:
            stick: 左摇杆或右摇杆
            x: X 轴值 (-32768 到 32767)
            y: Y 轴值 (-32768 到 32767)
            tilted: 倾斜持续时间（秒）
            released: 释放后等待时间（秒）
            block: 是否阻塞
        
        Returns:
            操作是否成功
        """
        logger.debug(f"tilt_stick: stick={stick}, x={x}, y={y}, tilted={tilted}, released={released}, block={block}")
        
        if not self.is_connected():
            logger.error("ESP32 未连接")
            return False
        
        # 将坐标转换为角度和强度
        # 归一化到 -1 到 1 范围
        x_norm = max(-1.0, min(1.0, x / 32767.0))
        y_norm = max(-1.0, min(1.0, y / 32767.0))
        
        # 计算角度（0-359度，0=右，90=上，180=左，270=下）
        import math
        angle = 0
        if not (x_norm == 0 and y_norm == 0):
            # atan2(y, x): 0=右, pi/2=上, pi/-pi=左, -pi/2=下
            angle_rad = math.atan2(y_norm, x_norm)
            angle = math.degrees(angle_rad)
            # 标准化到0-359范围
            angle = (angle + 360) % 360
        
        # 计算强度（0-255）
        intensity = int(min(255, ((x_norm**2 + y_norm**2) ** 0.5) * 255))
        
        # 选择摇杆
        if stick == Controller.Stick.L_STICK:
            lstick = ESP32InputEmulator.lstick_angle(int(angle), intensity)
            rstick = ESP32InputEmulator.RSTICK_CENTER
        else:  # R_STICK
            lstick = ESP32InputEmulator.LSTICK_CENTER
            rstick = ESP32InputEmulator.rstick_angle(int(angle), intensity)
        
        def _tilt():
            # 发送摇杆命令
            success = self._serial.send_buttons(
                buttons=[],
                dpad=ESP32InputEmulator.DPad.CENTER,
                lstick=lstick,
                rstick=rstick,
                temporary=False
            )
            
            if success and tilted > 0:
                time.sleep(tilted)
            
            # 重置摇杆到中心
            if success:
                success = self._serial.send_command(ESP32InputEmulator.NO_INPUT)
            
            if success and released > 0:
                time.sleep(released)
            
            return success
        
        if block:
            return _tilt()
        else:
            thread = threading.Thread(target=_tilt)
            thread.start()
            return True
    
    def macro(self, macro: str, block: bool = True) -> bool:
        """
        执行宏命令
        
        Args:
            macro: 宏命令字符串
            block: 是否阻塞
        
        Returns:
            操作是否成功
        """
        logger.debug(f"macro: macro长度={len(macro)}, block={block}")
        
        # 使用父类的默认实现，它会调用 press_buttons 和 tilt_stick
        return super().macro(macro, block)
    
    def __del__(self):
        """析构函数，确保断开连接"""
        try:
            self.disconnect()
        except:
            pass