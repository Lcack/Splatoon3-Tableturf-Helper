#!/usr/bin/env python3
"""
AutoTableTurf Qt 桌面应用主入口
"""

import sys
import os
import traceback

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon

from qt.home.main_window import MainWindow
from qt.debug.debug_window import DebugWindow
from logger import logger


class AutoTableTurfApp(QMainWindow):
    """主应用窗口，包含标签页"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AutoTableTurf - Qt 桌面版")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用图标（如果有）
        icon_path = os.path.join(os.path.dirname(__file__), "icons", "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # 创建主窗口
        self.main_window = MainWindow()
        self.tab_widget.addTab(self.main_window, "主控制")
        
        # 创建调试窗口
        self.debug_window = DebugWindow()
        self.tab_widget.addTab(self.debug_window, "调试")
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #0078d7;
            }
        """)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
        
        # 定时器用于更新状态
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # 每秒更新一次
        
        logger.info("AutoTableTurf Qt 应用已启动")
    
    def update_status(self):
        """更新状态栏"""
        # 这里可以添加状态更新逻辑
        pass
    
    def closeEvent(self, event):
        """关闭事件处理"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出 AutoTableTurf 吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清理资源
            self.main_window.cleanup()
            self.debug_window.cleanup()
            
            logger.info("AutoTableTurf Qt 应用正在关闭")
            event.accept()
        else:
            event.ignore()


def main():
    """主函数"""
    try:
        # 创建应用
        app = QApplication(sys.argv)
        app.setApplicationName("AutoTableTurf")
        app.setOrganizationName("AutoTableTurf")
        
        # 设置全局样式
        app.setStyle("Fusion")
        
        # 创建主窗口
        window = AutoTableTurfApp()
        window.show()
        
        # 运行应用
        logger.info("启动 Qt 应用...")
        return_code = app.exec()
        
        logger.info(f"Qt 应用退出，返回码: {return_code}")
        sys.exit(return_code)
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        logger.error(traceback.format_exc())
        
        # 显示错误对话框
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setText("应用启动失败")
        error_msg.setInformativeText(str(e))
        error_msg.setWindowTitle("错误")
        error_msg.exec()
        
        sys.exit(1)


if __name__ == "__main__":
    main()