#!/usr/bin/env python3
"""
启动 Portal 网页服务器
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portal import app

if __name__ == '__main__':
    print("=" * 50)
    print("AutoTableTurf Portal 启动中...")
    print("=" * 50)
    print(f"访问地址: http://localhost:5000")
    print(f"访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    # 启动 Flask 服务器
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)