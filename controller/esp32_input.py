"""
ESP32 输入模拟器模块
基于 AutoSplatoon 的 inputemulator.h 实现
"""

from enum import IntFlag
from typing import Tuple


class ESP32InputEmulator:
    """ESP32 输入模拟器，处理手柄按钮和摇杆的编码"""
    
    # 按钮位掩码（基于 AutoSplatoon 的 BTN_* 常量）
    class Button(IntFlag):
        NONE = 0x0000000000000000
        Y = 0x0000000000000001
        B = 0x0000000000000002
        A = 0x0000000000000004
        X = 0x0000000000000008
        L = 0x0000000000000010
        R = 0x0000000000000020
        ZL = 0x0000000000000040
        ZR = 0x0000000000000080
        MINUS = 0x0000000000000100
        PLUS = 0x0000000000000200
        L_CLICK = 0x0000000000000400  # L3
        R_CLICK = 0x0000000000000800  # R3
        HOME = 0x0000000000001000
        CAPTURE = 0x0000000000002000
        SL = 0x0000000000004000
        SR = 0x0000000000008000
    
    # 方向键位掩码
    class DPad(IntFlag):
        CENTER = 0x0000000000000000
        UP = 0x0000000000010000
        RIGHT = 0x0000000000020000
        DOWN = 0x0000000000040000
        LEFT = 0x0000000000080000
        UP_RIGHT = UP | RIGHT
        DOWN_RIGHT = DOWN | RIGHT
        UP_LEFT = UP | LEFT
        DOWN_LEFT = DOWN | LEFT
    
    # 左摇杆方向（角度和强度）
    @staticmethod
    def lstick_angle(angle: int, intensity: int = 255) -> int:
        """
        将角度和强度转换为左摇杆值
        angle: 0-359 度 (0=右, 90=上, 180=左, 270=下)
        intensity: 0-255 强度
        """
        if intensity == 0:
            return 0x0000000000000000
        
        angle = angle % 360
        angle_hex = angle << 16  # 角度放在高位
        intensity_hex = intensity << 24  # 强度放在更高位
        return 0xFF000000 | intensity_hex | angle_hex
    
    # 右摇杆方向（角度和强度）
    @staticmethod
    def rstick_angle(angle: int, intensity: int = 255) -> int:
        """
        将角度和强度转换为右摇杆值
        angle: 0-359 度 (0=右, 90=上, 180=左, 270=下)
        intensity: 0-255 强度
        """
        if intensity == 0:
            return 0x0000000000000000
        
        angle = angle % 360
        angle_hex = angle << 40  # 角度放在高位
        intensity_hex = intensity << 48  # 强度放在更高位
        return 0xFF00000000000000 | intensity_hex | angle_hex
    
    # 预定义的摇杆方向（基于 AutoSplatoon）
    LSTICK_CENTER = 0x0000000000000000
    LSTICK_RIGHT = 0x00000000FF000000      # 0度
    LSTICK_UP_RIGHT = 0x0000002DFF000000   # 45度
    LSTICK_UP = 0x0000005AFF000000         # 90度
    LSTICK_UP_LEFT = 0x00000087FF000000    # 135度
    LSTICK_LEFT = 0x000000B4FF000000       # 180度
    LSTICK_DOWN_LEFT = 0x000000E1FF000000  # 225度
    LSTICK_DOWN = 0x0000010EFF000000       # 270度
    LSTICK_DOWN_RIGHT = 0x0000013BFF000000 # 315度
    
    RSTICK_CENTER = 0x0000000000000000
    RSTICK_RIGHT = 0x000FF00000000000      # 0度
    RSTICK_UP_RIGHT = 0x02DFF00000000000   # 45度
    RSTICK_UP = 0x05AFF00000000000         # 90度
    RSTICK_UP_LEFT = 0x087FF00000000000    # 135度
    RSTICK_LEFT = 0x0B4FF00000000000       # 180度
    RSTICK_DOWN_LEFT = 0x0E1FF00000000000  # 225度
    RSTICK_DOWN = 0x10EFF00000000000       # 270度
    RSTICK_DOWN_RIGHT = 0x13BFF00000000000 # 315度
    
    NO_INPUT = Button.NONE | DPad.CENTER | LSTICK_CENTER | RSTICK_CENTER
    
    @staticmethod
    def crc8_ccitt(old_crc: int, new_data: int) -> int:
        """
        CRC-8-CCITT 校验计算
        基于 AutoSplatoon 的 _crc8 方法
        """
        data = old_crc ^ new_data
        for _ in range(8):
            if data & 0x80:
                data = (data << 1) ^ 0x07
            else:
                data = data << 1
            data &= 0xFF
        return data
    
    @staticmethod
    def decrypt_dpad(dpad: int) -> int:
        """
        解密方向键值
        基于 AutoSplatoon 的 _decrypt_dpad 方法
        """
        # AutoSplatoon 中的映射关系
        dpad_map = {
            0x08: DPad.CENTER,
            0x00: DPad.UP,
            0x01: DPad.UP_RIGHT,
            0x02: DPad.RIGHT,
            0x03: DPad.DOWN_RIGHT,
            0x04: DPad.DOWN,
            0x05: DPad.DOWN_LEFT,
            0x06: DPad.LEFT,
            0x07: DPad.UP_LEFT,
        }
        return dpad_map.get(dpad, DPad.CENTER)
    
    @staticmethod
    def cmd_to_packet(command: int) -> bytes:
        """
        将命令转换为数据包
        基于 AutoSplatoon 的 _cmd_to_packet 方法
        
        数据包格式（8字节）:
        [按钮低字节, 按钮高字节, 左摇杆X, 左摇杆Y, 右摇杆X, 右摇杆Y, HAT, CRC]
        """
        # 提取按钮部分（低16位）
        buttons_low = (command & 0xFF)
        buttons_high = ((command >> 8) & 0xFF)
        
        # 提取方向键（16-19位）
        dpad = (command >> 16) & 0x0F
        
        # 提取左摇杆（24-55位）
        lstick_value = (command >> 24) & 0xFFFFFFFF
        lstick_x = ((lstick_value >> 16) & 0xFF) if lstick_value else 0x80  # 中心值
        lstick_y = (lstick_value >> 24) & 0xFF if lstick_value else 0x80    # 中心值
        
        # 提取右摇杆（56-87位）
        rstick_value = (command >> 56) & 0xFFFFFFFF
        rstick_x = ((rstick_value >> 16) & 0xFF) if rstick_value else 0x80  # 中心值
        rstick_y = (rstick_value >> 24) & 0xFF if rstick_value else 0x80    # 中心值
        
        # HAT（方向键）值转换
        hat_map = {
            DPad.CENTER: 0x08,
            DPad.UP: 0x00,
            DPad.UP_RIGHT: 0x01,
            DPad.RIGHT: 0x02,
            DPad.DOWN_RIGHT: 0x03,
            DPad.DOWN: 0x04,
            DPad.DOWN_LEFT: 0x05,
            DPad.LEFT: 0x06,
            DPad.UP_LEFT: 0x07,
        }
        hat = hat_map.get(dpad, 0x08)
        
        # 构建数据包
        packet = bytearray([
            buttons_low,
            buttons_high,
            lstick_x,
            lstick_y,
            rstick_x,
            rstick_y,
            hat,
            0x00  # CRC 占位
        ])
        
        # 计算 CRC
        crc = 0
        for byte in packet[:-1]:
            crc = ESP32InputEmulator.crc8_ccitt(crc, byte)
        packet[-1] = crc
        
        return bytes(packet)