"""
ESP32 串口控制器模块
基于 AutoSplatoon 的 serialcontroller.cpp 实现
"""

import serial
import serial.tools.list_ports
import time
import threading
from typing import Optional, List
from enum import Enum

from controller.esp32_input import ESP32InputEmulator


class SerialStatus(Enum):
    """串口状态枚举"""
    NOT_CONNECTED = 0
    CONNECTED = 1
    SYNCING = 2
    SYNCED = 3
    ERROR = 4


class ESP32SerialController:
    """
    ESP32 串口控制器
    处理与 ESP32 的串口通信
    """
    
    # 波特率（基于 AutoSplatoon）
    BAUD_RATE = 19200
    
    # 同步字节（基于 AutoSplatoon）
    VANILLA_SYNC_BYTES = bytes([0xFF, 0x33, 0xCC])
    VANILLA_SYNC_RESP = bytes([0xFF, 0xCC, 0x33])
    
    CHOCOLATE_SYNC_BYTES = bytes([0xFF, 0x44, 0xEE])
    CHOCOLATE_SYNC_RESP = bytes([0xFF, 0xEE])
    
    # 响应代码
    RESP_USB_ACK = 0x90
    RESP_UPDATE_ACK = 0x91
    RESP_UPDATE_NACK = 0x92
    
    def __init__(self):
        self._serial_port: Optional[serial.Serial] = None
        self._port_name: Optional[str] = None
        self._status = SerialStatus.NOT_CONNECTED
        self._sync_state = 0  # 0=未同步, 1=同步中, 2=已同步
        self._lock = threading.RLock()
        self._input_emulator = ESP32InputEmulator()
        
        # 空白数据包（所有值居中）
        self._blank_packet = bytes([0x00, 0x00, 0x08, 0x80, 0x80, 0x80, 0x80, 0x00])
    
    def list_ports(self) -> List[str]:
        """列出可用的串口"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def open_and_sync(self, port_name: str, timeout: float = 5.0) -> bool:
        """
        打开串口并尝试同步
        基于 AutoSplatoon 的 openAndSync 方法
        """
        with self._lock:
            if self._serial_port and self._serial_port.is_open:
                self.close()
            
            try:
                self._serial_port = serial.Serial(
                    port=port_name,
                    baudrate=self.BAUD_RATE,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=timeout,
                    write_timeout=timeout
                )
                self._port_name = port_name
                self._status = SerialStatus.CONNECTED
                
                # 尝试同步
                return self._sync()
                
            except Exception as e:
                print(f"打开串口失败: {e}")
                self._status = SerialStatus.ERROR
                return False
    
    def close(self):
        """关闭串口连接"""
        with self._lock:
            if self._serial_port and self._serial_port.is_open:
                try:
                    # 发送空白数据包停止所有输入
                    self._send_packet(self._blank_packet)
                    time.sleep(0.1)
                    self._serial_port.close()
                except:
                    pass
                finally:
                    self._serial_port = None
                    self._port_name = None
                    self._status = SerialStatus.NOT_CONNECTED
                    self._sync_state = 0
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._status == SerialStatus.CONNECTED or self._status == SerialStatus.SYNCED
    
    def is_synced(self) -> bool:
        """检查是否已同步"""
        return self._sync_state == 2
    
    def get_status(self) -> SerialStatus:
        """获取当前状态"""
        return self._status
    
    def send_command(self, command: int, temporary: bool = False) -> bool:
        """
        发送命令到 ESP32
        基于 AutoSplatoon 的 recieveButtonAction 方法
        
        command: 使用 ESP32InputEmulator 生成命令
        temporary: 是否为临时按钮（按下后立即释放）
        """
        with self._lock:
            if not self.is_connected() or not self.is_synced():
                print("未连接或未同步")
                return False
            
            try:
                # 将命令转换为数据包
                packet = self._input_emulator.cmd_to_packet(command)
                
                # 发送数据包
                if not self._send_packet(packet):
                    return False
                
                # 如果是临时按钮，发送空白数据包
                if temporary:
                    time.sleep(0.05)  # 短暂延迟
                    if not self._send_packet(self._blank_packet):
                        return False
                
                return True
                
            except Exception as e:
                print(f"发送命令失败: {e}")
                return False
    
    def send_buttons(self, buttons: List[ESP32InputEmulator.Button], 
                     dpad: ESP32InputEmulator.DPad = ESP32InputEmulator.DPad.CENTER,
                     lstick: int = ESP32InputEmulator.LSTICK_CENTER,
                     rstick: int = ESP32InputEmulator.RSTICK_CENTER,
                     temporary: bool = False) -> bool:
        """
        发送按钮组合命令
        """
        # 组合按钮
        button_value = 0
        for button in buttons:
            button_value |= button.value
        
        # 组合完整命令
        command = button_value | dpad.value | lstick | rstick
        return self.send_command(command, temporary)
    
    def _sync(self) -> bool:
        """
        与 ESP32 同步
        基于 AutoSplatoon 的同步逻辑
        """
        with self._lock:
            if not self._serial_port or not self._serial_port.is_open:
                return False
            
            self._status = SerialStatus.SYNCING
            self._sync_state = 1
            
            try:
                # 清空缓冲区
                self._serial_port.reset_input_buffer()
                self._serial_port.reset_output_buffer()
                
                # 尝试 Vanilla 同步
                print("尝试 Vanilla 同步...")
                self._serial_port.write(self.VANILLA_SYNC_BYTES)
                self._serial_port.flush()
                
                # 读取响应
                response = self._serial_port.read(3)
                if response == self.VANILLA_SYNC_RESP:
                    print("Vanilla 同步成功")
                    self._sync_state = 2
                    self._status = SerialStatus.SYNCED
                    return True
                
                # 尝试 Chocolate 同步
                print("尝试 Chocolate 同步...")
                self._serial_port.reset_input_buffer()
                self._serial_port.write(self.CHOCOLATE_SYNC_BYTES)
                self._serial_port.flush()
                
                # 读取响应
                response = self._serial_port.read(3)
                if len(response) >= 2 and response[:2] == self.CHOCOLATE_SYNC_RESP:
                    print("Chocolate 同步成功")
                    self._sync_state = 2
                    self._status = SerialStatus.SYNCED
                    return True
                
                print("同步失败")
                self._sync_state = 0
                self._status = SerialStatus.CONNECTED
                return False
                
            except Exception as e:
                print(f"同步过程中出错: {e}")
                self._sync_state = 0
                self._status = SerialStatus.ERROR
                return False
    
    def _send_packet(self, packet: bytes) -> bool:
        """
        发送数据包到串口
        基于 AutoSplatoon 的 _sendPacket 方法
        """
        if not self._serial_port or not self._serial_port.is_open:
            return False
        
        try:
            self._serial_port.write(packet)
            self._serial_port.flush()
            return True
        except Exception as e:
            print(f"发送数据包失败: {e}")
            return False
    
    def _read_response(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        读取响应
        """
        if not self._serial_port or not self._serial_port.is_open:
            return None
        
        try:
            self._serial_port.timeout = timeout
            response = self._serial_port.read(1)
            return response if response else None
        except:
            return None