import datetime
import io
from typing import Optional

import cv2
import numpy as np
from flask import Response, request, render_template

from capture import VideoCapture
from controller import DummyController, Controller, NxbtController, Esp32Controller
import serial.tools.list_ports
from logger import logger
from portal.debug.debugger import web_debugger
from portal.home.capture import ThreadSafeCapture
from portal.home.closer import WebCloser
from portal.home.keymap import keymap
from tableturf.ai.alpha import Alpha
from tableturf.manager import TableTurfManager, Profile


def list_available_source():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


available_sources = list_available_source()
capture = ThreadSafeCapture(VideoCapture(0))
controller: Optional[Controller] = None
closer: WebCloser = None


def list_available_ports():
    """
    列出所有可用的串口端口
    """
    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append(port.device)
    return ports


def main():
    global available_sources
    available_sources = list_available_source()
    available_ports = list_available_ports()
    return Response(render_template(
        'home.html',
        url=request.url,
        sources=available_sources,
        ports=available_ports,
        keymap=keymap
    ))


def run():
    global closer
    debug = request.json['debug']
    sleep = request.json['sleep']
    profile = request.json['profile']
    stop_at = request.json['stop_at']
    try:
        time = datetime.datetime.fromisoformat(stop_at)
    except:
        time = None
    logger.debug(f'portal.home.run: profile={profile}, sleep={sleep}, debug={debug}, stop_at={stop_at}')
    alpha_ai = Alpha()
    manager = TableTurfManager(
        capture,
        controller if controller is not None else DummyController(),
        alpha_ai,
        web_debugger,
    )
    if closer is None:
        closer = WebCloser(time)
    manager.run(profile=Profile.from_json(profile), closer=closer, debug=debug)
    closer = None
    if sleep:
        controller.press_buttons([Controller.Button.HOME], down=3)
        controller.press_buttons([Controller.Button.A])
    return Response()


def stop():
    if closer is not None:
        closer.set_close()
    return Response()


def change_source():
    global capture
    source = int(request.json['source'])
    capture.update_capture(VideoCapture(source))
    logger.debug(f'portal.home.source_on_change: source={source}')
    return Response()


def generate_frames():
    empty = np.zeros((1080, 1920, 3))
    _, empty_buf = cv2.imencode(".jpeg", empty)
    empty_frame = bytes(io.BytesIO(empty_buf).getbuffer())
    while True:
        try:
            img = capture.capture()
            _, buffer = cv2.imencode(".jpeg", img)
            frame = bytes(io.BytesIO(buffer).getbuffer())
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        except Exception:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + empty_frame + b'\r\n'


def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def key_press():
    if controller is None:
        return Response()
    event_type = request.json['type']
    if event_type == 'keydown':
        raw = request.json['key']
        key = keymap.get(raw, None)
        if key is not None:
            if key == Controller.Stick.L_STICK:
                if raw == 'a':
                    controller.tilt_stick(key, -100, 0, tilted=0.3)
                elif raw == 's':
                    controller.tilt_stick(key, 0, -100, tilted=0.3)
                elif raw == 'd':
                    controller.tilt_stick(key, 100, 0, tilted=0.3)
                elif raw == 'w':
                    controller.tilt_stick(key, 0, 100, tilted=0.3)
            else:
                controller.press_buttons([key])
    # TODO: support long press
    return Response()


def connect_controller():
    global controller
    endpoint = request.json['endpoint']
    controller_type = request.json.get('controller_type', 'nxbt')
    port = request.json.get('port', '')
    
    if controller_type == 'esp32' and port:
        controller = Esp32Controller(port)
    elif endpoint != '':
        controller = NxbtController(endpoint)
    else:
        controller = DummyController()
    return Response()


def flash_esp32():
    """
    烧录 ESP32 固件
    """
    import subprocess
    import os
    import sys
    
    # 获取 esptool.exe 和固件文件的路径
    # 假设它们在项目根目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    esptool_path = os.path.join(base_dir, 'esptool.exe')
    firmware_path = os.path.join(base_dir, 'PRO-UART0.bin')
    
    # 检查文件是否存在
    if not os.path.exists(esptool_path):
        logger.error(f'esptool.exe not found at {esptool_path}')
        return Response('esptool.exe not found', status=500)
    
    if not os.path.exists(firmware_path):
        logger.error(f'Firmware file not found at {firmware_path}')
        return Response('Firmware file not found', status=500)
    
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
            return Response('烧录成功')
        else:
            logger.error('ESP32 flash failed')
            return Response('烧录失败', status=500)
            
    except subprocess.TimeoutExpired:
        logger.error('ESP32 flash timeout')
        return Response('烧录超时', status=500)
    except Exception as e:
        logger.error(f'ESP32 flash error: {e}')
        return Response(f'烧录错误: {str(e)}', status=500)