#!/usr/bin/env python3
"""
启动 Qt 桌面应用的便捷脚本
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qt import run_app

if __name__ == "__main__":
    run_app()