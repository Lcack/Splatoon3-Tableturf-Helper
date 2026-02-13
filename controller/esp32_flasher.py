import subprocess
import os
import sys
from logger import logger


def flash_esp32_firmware():
    """
    烧录 ESP32 固件
    
    Returns:
        tuple: (success, message) - 成功状态和消息
    """
    # 获取 esptool.exe 和固件文件的路径
    # 假设它们在项目根目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    esptool_path = os.path.join(base_dir, 'esptool.exe')
    firmware_path = os.path.join(base_dir, 'PRO-UART0.bin')
    
    # 检查文件是否存在
    if not os.path.exists(esptool_path):
        logger.error(f'esptool.exe not found at {esptool_path}')
        return False, 'esptool.exe not found'
    
    if not os.path.exists(firmware_path):
        logger.error(f'Firmware file not found at {firmware_path}')
        return False, 'Firmware file not found'
    
    try:
        # 构建烧录命令
        cmd = [
            esptool_path,
            '--baud', '230400',
            'write_flash',
            '0x0',
            firmware_path
        ]
        
        logger.info(f'Flashing ESP32 with command: {cmd}')
        
        # 执行命令并捕获输出
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 设置 60 秒超时
        )
        
        logger.info(f'Flash output: {result.stdout}')
        logger.info(f'Flash error: {result.stderr}')
        
        # 检查烧录是否成功
        if '100 %' in result.stdout and result.returncode == 0:
            logger.info('ESP32 flash successful')
            return True, '烧录成功'
        else:
            logger.error('ESP32 flash failed')
            return False, '烧录失败'
            
    except subprocess.TimeoutExpired:
        logger.error('ESP32 flash timeout')
        return False, '烧录超时'
    except Exception as e:
        logger.error(f'ESP32 flash error: {e}')
        return False, f'烧录错误: {str(e)}'