import time
import serial
from typing import List

from controller import Controller
from logger import logger


class Esp32Controller(Controller):
    def __init__(self, port, baudrate=19200, timeout=1):
        """
        初始化ESP32控制器
        :param port: 串口端口，例如 'COM3'
        :param baudrate: 波特率，默认为19200
        :param timeout: 串口超时时间，默认为1秒
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """
        连接到ESP32设备并执行握手协议
        """
        try:
            # 打开串口
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            
            # 执行巧克力模式握手
            self._handshake()
            self.connected = True
            logger.info(f'ESP32 controller connected to {self.port}')
        except Exception as e:
            logger.error(f'Failed to connect to ESP32: {e}')
            self.connected = False
    
    def _handshake(self):
        """
        执行握手协议
        """
        # 发送同步字节
        sync_bits = b'\xff' * 10
        self.serial.write(sync_bits)
        self.serial.flush()
        
        # 等待响应并执行巧克力模式握手
        # MASTER: 0xFF
        # SLAVE: 0xFF
        # MASTER: 0x44
        # SLAVE: 0xEE
        # MASTER: 0xEE
        # SLAVE: CONTROLLER_TYPE
        
        # 读取响应
        time.sleep(0.1)
        response = self.serial.read(1)
        if response != b'\xff':
            logger.warning(f'Unexpected handshake response: {response}')
        
        # 发送 0x44
        self.serial.write(b'\x44')
        self.serial.flush()
        
        # 读取 0xEE
        time.sleep(0.1)
        response = self.serial.read(1)
        if response != b'\xee':
            logger.warning(f'Unexpected handshake response: {response}')
        
        # 发送 0xEE
        self.serial.write(b'\xee')
        self.serial.flush()
        
        # 读取控制器类型
        time.sleep(0.1)
        controller_type = self.serial.read(1)
        logger.debug(f'ESP32 controller type: {controller_type}')
    
    def _crc8_ccitt_update(self, in_crc, in_data):
        """
        计算CRC8校验值
        :param in_crc: 初始CRC值
        :param in_data: 数据字节
        :return: 计算后的CRC值
        """
        data = in_crc ^ in_data
        for i in range(8):
            if (data & 0x80) != 0:
                data <<= 1
                data ^= 0x07
            else:
                data <<= 1
        return data
    
    def _calculate_crc(self, packet):
        """
        计算数据包的CRC校验值
        :param packet: 数据包（前8个字节）
        :return: CRC校验值
        """
        crc = 0
        for byte in packet:
            crc = self._crc8_ccitt_update(crc, byte)
        return crc
    
    def _build_packet(self, buttons, l_stick_x, l_stick_y, r_stick_x, r_stick_y):
        """
        构建控制器数据包
        :param buttons: 按钮状态
        :param l_stick_x: 左摇杆X值 (-100 to 100)
        :param l_stick_y: 左摇杆Y值 (-100 to 100)
        :param r_stick_x: 右摇杆X值 (-100 to 100)
        :param r_stick_y: 右摇杆Y值 (-100 to 100)
        :return: 完整的数据包（包含CRC）
        """
        # 初始化数据包
        packet = [0] * 8
        
        # 处理按钮状态
        # Byte 0: Y, B, A, X, L, R, ZL, ZR
        # Byte 1: Minus, Plus, L Clk, R Clk, Home, Capture, SL, SR
        
        # 映射按钮到字节
        button_map = {
            Controller.Button.Y: (0, 0),      # Byte 0, Bit 0
            Controller.Button.B: (0, 1),      # Byte 0, Bit 1
            Controller.Button.A: (0, 2),      # Byte 0, Bit 2
            Controller.Button.X: (0, 3),      # Byte 0, Bit 3
            Controller.Button.L: (0, 4),      # Byte 0, Bit 4
            Controller.Button.R: (0, 5),      # Byte 0, Bit 5
            Controller.Button.ZL: (0, 6),     # Byte 0, Bit 6
            Controller.Button.ZR: (0, 7),     # Byte 0, Bit 7
            Controller.Button.MINUS: (1, 0),  # Byte 1, Bit 0
            Controller.Button.PLUS: (1, 1),   # Byte 1, Bit 1
            Controller.Button.L_STICK_PRESS: (1, 2),  # Byte 1, Bit 2
            Controller.Button.R_STICK_PRESS: (1, 3),  # Byte 1, Bit 3
            Controller.Button.HOME: (1, 4),   # Byte 1, Bit 4
            Controller.Button.CAPTURE: (1, 5),# Byte 1, Bit 5
            Controller.Button.JCL_SL: (1, 6), # Byte 1, Bit 6 (SL)
            Controller.Button.JCL_SR: (1, 7), # Byte 1, Bit 7 (SR)
        }
        
        # 设置按钮状态
        for button in buttons:
            if button in button_map:
                byte_idx, bit_idx = button_map[button]
                packet[byte_idx] |= (1 << bit_idx)
        
        # 设置DPad状态（默认为中心）
        packet[2] = 0x08  # A_DPAD_CENTER
        
        # 设置摇杆值（范围从-100到100映射到0-255）
        packet[3] = int((l_stick_x + 100) * 255 / 200)  # LX
        packet[4] = int((l_stick_y + 100) * 255 / 200)  # LY
        packet[5] = int((r_stick_x + 100) * 255 / 200)  # RX
        packet[6] = int((r_stick_y + 100) * 255 / 200)  # RY
        
        # 计算CRC
        crc = self._calculate_crc(packet)
        packet.append(crc)
        
        return bytes(packet)
    
    def press_buttons(self, buttons: List[Controller.Button], down: float = 0.08, up: float = 0.08, block=True) -> bool:
        """
        按下按钮
        :param buttons: 要按下的按钮列表
        :param down: 按下持续时间（秒）
        :param up: 松开持续时间（秒）
        :param block: 是否阻塞直到操作完成
        :return: 操作是否成功
        """
        if not self.connected or not self.serial:
            logger.error('ESP32 controller not connected')
            return False
        
        try:
            # 发送按下按钮的数据包
            packet = self._build_packet(buttons, 0, 0, 0, 0)
            self.serial.write(packet)
            self.serial.flush()
            
            # 等待按下持续时间
            time.sleep(down)
            
            # 发送松开按钮的数据包
            packet = self._build_packet([], 0, 0, 0, 0)
            self.serial.write(packet)
            self.serial.flush()
            
            # 等待松开持续时间
            time.sleep(up)
            
            return True
        except Exception as e:
            logger.error(f'Failed to press buttons: {e}')
            return False
    
    def tilt_stick(self, stick: Controller.Stick, x: int, y: int, tilted: float = 0.1, released: float = 0.1, block=True) -> bool:
        """
        倾斜摇杆
        :param stick: 要倾斜的摇杆
        :param x: X轴值（-100到100）
        :param y: Y轴值（-100到100）
        :param tilted: 倾斜持续时间（秒）
        :param released: 松开持续时间（秒）
        :param block: 是否阻塞直到操作完成
        :return: 操作是否成功
        """
        if not self.connected or not self.serial:
            logger.error('ESP32 controller not connected')
            return False
        
        try:
            # 初始化摇杆值
            lx, ly, rx, ry = 0, 0, 0, 0
            
            # 设置指定摇杆的值
            if stick == Controller.Stick.L_STICK:
                lx, ly = x, y
            elif stick == Controller.Stick.R_STICK:
                rx, ry = x, y
            
            # 发送倾斜摇杆的数据包
            packet = self._build_packet([], lx, ly, rx, ry)
            self.serial.write(packet)
            self.serial.flush()
            
            # 等待倾斜持续时间
            time.sleep(tilted)
            
            # 发送松开摇杆的数据包
            packet = self._build_packet([], 0, 0, 0, 0)
            self.serial.write(packet)
            self.serial.flush()
            
            # 等待松开持续时间
            time.sleep(released)
            
            return True
        except Exception as e:
            logger.error(f'Failed to tilt stick: {e}')
            return False
    
    def close(self):
        """
        关闭串口连接
        """
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.connected = False
            logger.info('ESP32 controller disconnected')
    
    def __del__(self):
        """
        析构函数，确保关闭串口
        """
        self.close()