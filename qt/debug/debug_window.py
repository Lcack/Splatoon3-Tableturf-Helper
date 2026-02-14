"""
调试窗口模块
"""

import os
import json
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTextEdit, QPushButton, QLabel, QGroupBox,
    QTreeWidget, QTreeWidgetItem, QSplitter,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QTextCursor

from logger import logger


class DebugWindow(QWidget):
    """调试窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.log_data = []
        self.debug_data: Dict[str, Any] = {}
        
        self.init_ui()
        self.init_connections()
        
        # 定时器用于更新日志
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_logs)
        
        logger.debug("调试窗口初始化完成")
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        self.clear_logs_btn = QPushButton("清空日志")
        toolbar_layout.addWidget(self.clear_logs_btn)
        
        self.save_logs_btn = QPushButton("保存日志")
        toolbar_layout.addWidget(self.save_logs_btn)
        
        self.refresh_btn = QPushButton("刷新")
        toolbar_layout.addWidget(self.refresh_btn)
        
        toolbar_layout.addStretch()
        
        self.auto_refresh_checkbox = QPushButton("自动刷新: 关")
        self.auto_refresh_checkbox.setCheckable(True)
        toolbar_layout.addWidget(self.auto_refresh_checkbox)
        
        layout.addLayout(toolbar_layout)
        
        # 主内容区域
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：日志显示
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333333;
                border-radius: 3px;
            }
        """)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        splitter.addWidget(log_group)
        
        # 右侧：调试信息
        debug_group = QGroupBox("调试信息")
        debug_layout = QVBoxLayout()
        
        # 调试数据树
        self.debug_tree = QTreeWidget()
        self.debug_tree.setHeaderLabels(["名称", "值"])
        self.debug_tree.setColumnWidth(0, 200)
        
        debug_layout.addWidget(self.debug_tree)
        
        # 原始数据查看
        self.raw_data_text = QTextEdit()
        self.raw_data_text.setFont(QFont("Monospace", 9))
        self.raw_data_text.setReadOnly(True)
        self.raw_data_text.setMaximumHeight(200)
        self.raw_data_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        
        debug_layout.addWidget(QLabel("原始数据:"))
        debug_layout.addWidget(self.raw_data_text)
        
        debug_group.setLayout(debug_layout)
        
        splitter.addWidget(debug_group)
        
        # 设置分割器比例
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def init_connections(self):
        """初始化信号连接"""
        self.clear_logs_btn.clicked.connect(self.clear_logs)
        self.save_logs_btn.clicked.connect(self.save_logs)
        self.refresh_btn.clicked.connect(self.refresh_debug_data)
        self.auto_refresh_checkbox.toggled.connect(self.toggle_auto_refresh)
        
        # 调试树选择变化
        self.debug_tree.itemSelectionChanged.connect(self.on_debug_item_selected)
    
    def clear_logs(self):
        """清空日志"""
        self.log_text.clear()
        self.log_data = []
        self.status_label.setText("日志已清空")
        logger.debug("调试日志已清空")
    
    def save_logs(self):
        """保存日志到文件"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存日志", 
                os.path.expanduser("~/autotableturf_debug.log"),
                "日志文件 (*.log);;所有文件 (*.*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                
                self.status_label.setText(f"日志已保存到: {file_path}")
                logger.info(f"调试日志已保存到: {file_path}")
                
        except Exception as e:
            self.status_label.setText(f"保存失败: {e}")
            logger.error(f"保存日志失败: {e}")
    
    def refresh_debug_data(self):
        """刷新调试数据"""
        try:
            # 这里可以添加获取调试数据的逻辑
            # 例如：从运行中的管理器获取调试信息
            
            # 模拟数据
            self.debug_data = {
                "system": {
                    "status": "running",
                    "uptime": "00:15:30",
                    "memory_usage": "45%"
                },
                "capture": {
                    "source": 0,
                    "resolution": "1280x720",
                    "fps": 30
                },
                "controller": {
                    "type": "dummy",
                    "connected": True
                },
                "ai": {
                    "state": "idle",
                    "last_action": "press A",
                    "confidence": 0.85
                }
            }
            
            # 更新调试树
            self.update_debug_tree()
            
            # 更新原始数据
            self.raw_data_text.setPlainText(
                json.dumps(self.debug_data, indent=2, ensure_ascii=False)
            )
            
            self.status_label.setText("调试数据已刷新")
            
        except Exception as e:
            self.status_label.setText(f"刷新失败: {e}")
            logger.error(f"刷新调试数据失败: {e}")
    
    def update_debug_tree(self):
        """更新调试树"""
        self.debug_tree.clear()
        
        def add_tree_items(parent: Optional[QTreeWidgetItem], data: Any, key: str = ""):
            """递归添加树项"""
            if isinstance(data, dict):
                for k, v in data.items():
                    item = QTreeWidgetItem(parent, [str(k), ""])
                    add_tree_items(item, v, k)
                    if parent is None:
                        self.debug_tree.addTopLevelItem(item)
            elif isinstance(data, list):
                for i, v in enumerate(data):
                    item = QTreeWidgetItem(parent, [f"[{i}]", ""])
                    add_tree_items(item, v, f"{key}[{i}]")
            else:
                if parent:
                    parent.setText(1, str(data))
        
        add_tree_items(None, self.debug_data)
        self.debug_tree.expandAll()
    
    def on_debug_item_selected(self):
        """调试树项选择变化"""
        selected_items = self.debug_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            self.status_label.setText(f"选中: {item.text(0)} = {item.text(1)}")
    
    def toggle_auto_refresh(self, checked: bool):
        """切换自动刷新"""
        if checked:
            self.log_timer.start(1000)  # 每秒更新一次
            self.auto_refresh_checkbox.setText("自动刷新: 开")
            self.status_label.setText("自动刷新已开启")
        else:
            self.log_timer.stop()
            self.auto_refresh_checkbox.setText("自动刷新: 关")
            self.status_label.setText("自动刷新已关闭")
    
    def update_logs(self):
        """更新日志显示"""
        # 这里可以添加从 logger 获取最新日志的逻辑
        # 目前只是模拟更新
        
        # 获取最新的日志（模拟）
        import time
        new_log = f"[{time.strftime('%H:%M:%S')}] 调试信息更新\n"
        
        # 添加到文本区域
        self.log_text.moveCursor(QTextCursor.End)
        self.log_text.insertPlainText(new_log)
        
        # 自动滚动到底部
        self.log_text.ensureCursorVisible()
        
        # 限制日志行数
        lines = self.log_text.toPlainText().split('\n')
        if len(lines) > 1000:
            self.log_text.setPlainText('\n'.join(lines[-500:]))
    
    def add_log_message(self, message: str, level: str = "INFO"):
        """添加日志消息"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        
        # 根据级别设置颜色
        color_map = {
            "DEBUG": "#888888",
            "INFO": "#ffffff",
            "WARNING": "#ffaa00",
            "ERROR": "#ff4444",
            "CRITICAL": "#ff0000"
        }
        
        color = color_map.get(level, "#ffffff")
        formatted_message = f'<font color="{color}">[{timestamp}] [{level}] {message}</font><br>'
        
        # 添加到文本区域
        self.log_text.moveCursor(QTextCursor.End)
        self.log_text.insertHtml(formatted_message)
        
        # 自动滚动到底部
        self.log_text.ensureCursorVisible()
        
        # 保存到数据
        self.log_data.append({
            "timestamp": timestamp,
            "level": level,
            "message": message
        })
        
        # 限制日志数据大小
        if len(self.log_data) > 1000:
            self.log_data = self.log_data[-500:]
    
    def set_debug_data(self, data: Dict[str, Any]):
        """设置调试数据"""
        self.debug_data = data
        self.update_debug_tree()
        self.raw_data_text.setPlainText(
            json.dumps(data, indent=2, ensure_ascii=False)
        )
    
    def cleanup(self):
        """清理资源"""
        self.log_timer.stop()
        logger.debug("调试窗口资源已清理")