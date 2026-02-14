import datetime
import io
import os
import subprocess
import threading
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np
from flask import Response, request, render_template, jsonify

from capture import VideoCapture
from controller import DummyController, Controller, NxbtController, ESP32Controller
from controller.esp32_flasher import flash_esp32 as flash_esp32_firmware, get_flash_status as get_esp32_flash_status, cancel_flash
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


def main():
    global available_sources
    available_sources = list_available_source()
    return Response(render_template(
        'home_esp32.html',  # 使用新的模板
        url=request.url,
        sources=available_sources,
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
    if endpoint != '':
        controller = NxbtController(endpoint)
    else:
        controller = DummyController()
    return Response()


# ESP32 相关变量
esp32_controller: Optional[ESP32Controller] = None


def get_esp32_ports():
    """获取可用的 ESP32 串口"""
    try:
        if esp32_controller is None:
            temp_controller = ESP32Controller(auto_connect=False)
        else:
            temp_controller = esp32_controller
        
        ports = temp_controller.list_available_ports()
        return jsonify({"ports": ports})
    except Exception as e:
        logger.error(f"获取串口失败: {e}")
        return jsonify({"ports": [], "error": str(e)})


def connect_esp32():
    """连接 ESP32"""
    global controller, esp32_controller
    
    port_name = request.json.get('port')
    if not port_name:
        return jsonify({"success": False, "message": "未指定串口"})
    
    try:
        # 断开现有连接
        if esp32_controller is not None:
            try:
                esp32_controller.disconnect()
            except:
                pass
        
        # 创建新连接
        esp32_controller = ESP32Controller(port_name=port_name, auto_connect=True)
        
        if esp32_controller.is_connected():
            controller = esp32_controller
            return jsonify({
                "success": True, 
                "message": f"ESP32 连接成功: {port_name}",
                "status": "connected"
            })
        else:
            return jsonify({
                "success": False, 
                "message": f"ESP32 连接失败: {port_name}",
                "status": "disconnected"
            })
            
    except Exception as e:
        logger.error(f"连接 ESP32 失败: {e}")
        return jsonify({
            "success": False, 
            "message": f"连接失败: {str(e)}",
            "status": "error"
        })


def disconnect_esp32():
    """断开 ESP32 连接"""
    global controller, esp32_controller
    
    try:
        if esp32_controller is not None:
            esp32_controller.disconnect()
            controller = DummyController()  # 切换回虚拟控制器
            return jsonify({
                "success": True, 
                "message": "ESP32 已断开连接",
                "status": "disconnected"
            })
        else:
            return jsonify({
                "success": False, 
                "message": "ESP32 未连接",
                "status": "disconnected"
            })
    except Exception as e:
        logger.error(f"断开 ESP32 连接失败: {e}")
        return jsonify({
            "success": False, 
            "message": f"断开连接失败: {str(e)}",
            "status": "error"
        })


def get_esp32_status():
    """获取 ESP32 状态"""
    if esp32_controller is None:
        return jsonify({
            "connected": False,
            "port": None,
            "status": "disconnected"
        })
    
    try:
        connected = esp32_controller.is_connected()
        return jsonify({
            "connected": connected,
            "port": esp32_controller._port_name if hasattr(esp32_controller, '_port_name') else None,
            "status": "connected" if connected else "disconnected"
        })
    except Exception as e:
        logger.error(f"获取 ESP32 状态失败: {e}")
        return jsonify({
            "connected": False,
            "port": None,
            "status": "error",
            "error": str(e)
        })


def flash_esp32():
    """烧录 ESP32 固件"""
    port_name = request.json.get('port')
    firmware_path = request.json.get('firmware_path', '')
    
    if not port_name:
        return jsonify({"success": False, "message": "未指定串口"})
    
    # 使用烧录器模块
    success = flash_esp32_firmware(port_name, firmware_path if firmware_path else None)
    
    if success:
        return jsonify({
            "success": True, 
            "message": "烧录已开始",
            "task_id": str(time.time())
        })
    else:
        return jsonify({
            "success": False, 
            "message": "已有烧录任务在进行中"
        })


def get_flash_status():
    """获取烧录状态"""
    status = get_esp32_flash_status()
    return jsonify(status)
