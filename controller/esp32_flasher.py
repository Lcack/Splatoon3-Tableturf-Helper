"""
ESP32 烧录工具模块
基于 AutoSplatoon 的烧录功能实现
"""

import os
import subprocess
import threading
import time
import serial.tools.list_ports
from typing import Optional, List, Dict, Any
from enum import Enum


class FlashStatus(Enum):
    """烧录状态枚举"""
    IDLE = "idle"
    FLASHING = "flashing"
    SUCCESS = "success"
    FAILED = "failed"


class ESP32Flasher:
    """
    ESP32 烧录工具
    使用 esptool.py 烧录固件到 ESP32
    """
    
    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._output = ""
        self._status = FlashStatus.IDLE
        self._progress = 0
        self._lock = threading.RLock()
        self._flash_thread: Optional[threading.Thread] = None
        
    def list_ports(self) -> List[str]:
        """列出可用的串口"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def is_flashing(self) -> bool:
        """检查是否正在烧录"""
        with self._lock:
            return self._status == FlashStatus.FLASHING
    
    def get_status(self) -> Dict[str, Any]:
        """获取烧录状态"""
        with self._lock:
            return {
                "status": self._status.value,
                "progress": self._progress,
                "output": self._output,
                "running": self._status == FlashStatus.FLASHING
            }
    
    def flash(self, port: str, firmware_path: Optional[str] = None, 
              baud_rate: int = 230400, flash_mode: str = "dio", 
              flash_size: str = "4MB") -> bool:
        """
        烧录固件到 ESP32
        
        Args:
            port: 串口名称
            firmware_path: 固件文件路径，如果为 None 则使用默认固件
            baud_rate: 波特率
            flash_mode: 闪存模式
            flash_size: 闪存大小
        
        Returns:
            是否成功启动烧录
        """
        with self._lock:
            if self._status == FlashStatus.FLASHING:
                return False
            
            # 重置状态
            self._output = ""
            self._progress = 0
            self._status = FlashStatus.FLASHING
            
            # 启动烧录线程
            self._flash_thread = threading.Thread(
                target=self._flash_thread_func,
                args=(port, firmware_path, baud_rate, flash_mode, flash_size),
                daemon=True
            )
            self._flash_thread.start()
            
            return True
    
    def _flash_thread_func(self, port: str, firmware_path: Optional[str], 
                          baud_rate: int, flash_mode: str, flash_size: str):
        """烧录线程函数"""
        try:
            # 构建 esptool 命令
            cmd = [
                'esptool.py',
                '--port', port,
                '--baud', str(baud_rate),
                '--before', 'default_reset',
                '--after', 'hard_reset',
                'write_flash',
                '-z',
                '--flash_mode', flash_mode,
                '--flash_size', flash_size,
                '0x1000',  # bootloader 偏移地址
            ]
            
            # 添加固件路径
            if firmware_path and os.path.exists(firmware_path):
                cmd.append(firmware_path)
            else:
                # 尝试使用默认固件
                default_firmware = self._find_default_firmware()
                if default_firmware:
                    cmd.append(default_firmware)
                else:
                    with self._lock:
                        self._output = "错误：未找到固件文件\n"
                        self._status = FlashStatus.FAILED
                    return
            
            self._add_output(f"执行命令: {' '.join(cmd)}\n")
            
            # 执行烧录
            with self._lock:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
            
            # 读取输出
            for line in self._process.stdout:
                self._add_output(line)
                self._update_progress_from_output(line)
            
            # 等待进程结束
            return_code = self._process.wait()
            
            with self._lock:
                self._process = None
                
                if return_code == 0 and self._progress == 100:
                    self._status = FlashStatus.SUCCESS
                    self._add_output("\n✅ 烧录成功！\n")
                else:
                    self._status = FlashStatus.FAILED
                    self._add_output(f"\n❌ 烧录失败，返回码: {return_code}\n")
                    
        except Exception as e:
            with self._lock:
                self._output += f"\n烧录过程中出错: {str(e)}\n"
                self._status = FlashStatus.FAILED
                self._process = None
        finally:
            with self._lock:
                if self._process:
                    self._process.terminate()
                    self._process = None
    
    def _find_default_firmware(self) -> Optional[str]:
        """查找默认固件文件"""
        # 在当前目录下查找
        possible_paths = [
            "firmware/autotableturf.bin",
            "firmware/firmware.bin",
            "autotableturf.bin",
            "firmware.bin",
            "../firmware/autotableturf.bin",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        # 在 AutoSplatoon 目录中查找
        autosplatoon_path = os.path.expanduser("~/AutoSplatoon")
        if os.path.exists(autosplatoon_path):
            possible_autosplatoon_paths = [
                os.path.join(autosplatoon_path, "PRO-UART0.bin"),
                os.path.join(autosplatoon_path, "firmware.bin"),
            ]
            for path in possible_autosplatoon_paths:
                if os.path.exists(path):
                    return path
        
        return None
    
    def _add_output(self, text: str):
        """添加输出文本"""
        with self._lock:
            self._output += text
    
    def _update_progress_from_output(self, line: str):
        """从输出中更新进度"""
        import re
        
        # 查找进度百分比
        match = re.search(r'\((\d+) %\)', line)
        if match:
            progress = int(match.group(1))
            with self._lock:
                self._progress = progress
        
        # 查找写入进度
        match = re.search(r'Writing at 0x[0-9a-f]+\.\.\. \((\d+) %\)', line)
        if match:
            progress = int(match.group(1))
            with self._lock:
                self._progress = progress
    
    def cancel(self):
        """取消烧录"""
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                self._process = None
                self._status = FlashStatus.FAILED
                self._output += "\n烧录已取消\n"
                return True
        return False
    
    def __del__(self):
        """析构函数，确保清理进程"""
        try:
            self.cancel()
        except:
            pass


# 全局烧录器实例
_global_flasher: Optional[ESP32Flasher] = None


def get_flasher() -> ESP32Flasher:
    """获取全局烧录器实例"""
    global _global_flasher
    if _global_flasher is None:
        _global_flasher = ESP32Flasher()
    return _global_flasher


def flash_esp32(port: str, firmware_path: Optional[str] = None) -> bool:
    """
    烧录 ESP32 固件（便捷函数）
    
    Args:
        port: 串口名称
        firmware_path: 固件文件路径
    
    Returns:
        是否成功启动烧录
    """
    flasher = get_flasher()
    return flasher.flash(port, firmware_path)


def get_flash_status() -> Dict[str, Any]:
    """获取烧录状态"""
    flasher = get_flasher()
    return flasher.get_status()


def cancel_flash() -> bool:
    """取消烧录"""
    flasher = get_flasher()
    return flasher.cancel()