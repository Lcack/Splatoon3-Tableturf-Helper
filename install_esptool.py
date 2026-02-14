#!/usr/bin/env python3
"""
安装 esptool 脚本
用于烧录 ESP32 固件
"""

import subprocess
import sys
import os


def check_esptool():
    """检查是否已安装 esptool"""
    try:
        result = subprocess.run(
            ['esptool.py', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ esptool 已安装: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    return False


def install_esptool():
    """安装 esptool"""
    print("正在安装 esptool...")
    
    try:
        # 使用 pip 安装
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'esptool'],
            check=True
        )
        
        # 验证安装
        if check_esptool():
            print("✅ esptool 安装成功")
            return True
        else:
            print("❌ esptool 安装失败")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 安装过程中出错: {e}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("ESP32 烧录工具安装脚本")
    print("=" * 50)
    
    # 检查是否已安装
    if check_esptool():
        print("\nesptool 已安装，无需重复安装")
        return
    
    # 询问是否安装
    response = input("\n是否安装 esptool？(y/n): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("取消安装")
        return
    
    # 安装
    if install_esptool():
        print("\n安装完成！")
        print("\n使用说明:")
        print("1. 进入 ESP32 下载模式（按住 BOOT，点击 EN）")
        print("2. 通过 USB 连接 ESP32")
        print("3. 在网页中选择正确的串口")
        print("4. 点击烧录按钮")
    else:
        print("\n安装失败，请手动安装:")
        print("pip install esptool")


if __name__ == "__main__":
    main()