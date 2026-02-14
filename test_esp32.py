#!/usr/bin/env python3
"""
ESP32 控制器测试脚本
"""

import sys
import time
from controller.esp32 import ESP32Controller
from controller import Controller


def list_ports():
    """列出可用串口"""
    print("=== 可用串口 ===")
    controller = ESP32Controller(auto_connect=False)
    ports = controller.list_available_ports()
    
    if not ports:
        print("未找到可用串口")
    else:
        for i, port in enumerate(ports):
            print(f"{i+1}. {port}")
    
    return ports


def test_connection(port_name):
    """测试连接"""
    print(f"\n=== 测试连接: {port_name} ===")
    
    controller = ESP32Controller(port_name=port_name, auto_connect=True)
    
    if controller.is_connected():
        print("✅ 连接成功")
        return controller
    else:
        print("❌ 连接失败")
        return None


def test_buttons(controller):
    """测试按钮"""
    print("\n=== 测试按钮 ===")
    
    # 测试 A 按钮
    print("按下 A 按钮...")
    success = controller.press_buttons([Controller.Button.A], down=0.5, up=0.5)
    print(f"A 按钮: {'✅ 成功' if success else '❌ 失败'}")
    
    time.sleep(1)
    
    # 测试 B 按钮
    print("按下 B 按钮...")
    success = controller.press_buttons([Controller.Button.B], down=0.5, up=0.5)
    print(f"B 按钮: {'✅ 成功' if success else '❌ 失败'}")
    
    time.sleep(1)
    
    # 测试方向键
    print("按下 上 方向键...")
    success = controller.press_buttons([Controller.Button.DPAD_UP], down=0.5, up=0.5)
    print(f"上方向键: {'✅ 成功' if success else '❌ 失败'}")
    
    time.sleep(1)
    
    # 测试组合键
    print("按下 A+B 组合键...")
    success = controller.press_buttons([Controller.Button.A, Controller.Button.B], down=0.5, up=0.5)
    print(f"A+B 组合键: {'✅ 成功' if success else '❌ 失败'}")


def test_sticks(controller):
    """测试摇杆"""
    print("\n=== 测试摇杆 ===")
    
    # 测试左摇杆向右
    print("左摇杆向右...")
    success = controller.tilt_stick(Controller.Stick.L_STICK, x=20000, y=0, tilted=0.5, released=0.5)
    print(f"左摇杆向右: {'✅ 成功' if success else '❌ 失败'}")
    
    time.sleep(1)
    
    # 测试右摇杆向上
    print("右摇杆向上...")
    success = controller.tilt_stick(Controller.Stick.R_STICK, x=0, y=20000, tilted=0.5, released=0.5)
    print(f"右摇杆向上: {'✅ 成功' if success else '❌ 失败'}")


def main():
    """主函数"""
    print("ESP32 控制器测试")
    print("=" * 50)
    
    # 列出串口
    ports = list_ports()
    
    if not ports:
        print("\n请连接 ESP32 设备后重试")
        return
    
    # 选择串口
    if len(ports) == 1:
        port_name = ports[0]
        print(f"\n自动选择串口: {port_name}")
    else:
        try:
            choice = int(input(f"\n选择串口 (1-{len(ports)}): "))
            if 1 <= choice <= len(ports):
                port_name = ports[choice-1]
            else:
                print("无效选择")
                return
        except:
            print("无效输入")
            return
    
    # 测试连接
    controller = test_connection(port_name)
    if not controller:
        return
    
    try:
        # 测试按钮
        test_buttons(controller)
        
        # 测试摇杆
        test_sticks(controller)
        
        print("\n" + "=" * 50)
        print("✅ 所有测试完成")
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
    finally:
        # 断开连接
        print("\n断开连接...")
        controller.disconnect()


if __name__ == "__main__":
    main()