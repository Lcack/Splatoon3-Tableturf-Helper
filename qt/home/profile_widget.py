"""
配置文件部件 - 编辑和验证配置文件
"""

import json
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QColor

from logger import logger


class JsonHighlighter(QSyntaxHighlighter):
    """JSON 语法高亮"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 定义高亮规则
        self.highlighting_rules = []
        
        # 关键字格式
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(136, 19, 145))  # 紫色
        keyword_format.setFontWeight(QFont.Bold)
        
        keywords = ["true", "false", "null"]
        for word in keywords:
            pattern = r'\b' + word + r'\b'
            self.highlighting_rules.append((pattern, keyword_format))
        
        # 字符串格式
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(42, 161, 152))  # 青色
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))
        
        # 数字格式
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(28, 113, 216))  # 蓝色
        self.highlighting_rules.append((r'\b[0-9]+\b', number_format))
        
        # 键格式
        key_format = QTextCharFormat()
        key_format.setForeground(QColor(181, 137, 0))  # 金色
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"\s*:', key_format))
    
    def highlightBlock(self, text):
        """高亮文本块"""
        for pattern, format in self.highlighting_rules:
            import re
            for match in re.finditer(pattern, text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)


class ProfileWidget(QWidget):
    """配置文件编辑部件"""
    
    # 信号定义
    profile_changed = Signal(str)  # 传递配置文件文本
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.default_profile = """[
  {
    "current_level": 1,
    "current_win": 0,
    "target_level": 3,
    "target_win": 30,
    "deck": 0
  },
  {
    "current_level": 1,
    "current_win": 0,
    "target_level": 3,
    "target_win": 30,
    "deck": 0
  }
]"""
        
        self.init_ui()
        self.init_connections()
        
        logger.debug("配置文件部件初始化完成")
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout()
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        self.load_default_btn = QPushButton("加载默认")
        toolbar_layout.addWidget(self.load_default_btn)
        
        self.validate_btn = QPushButton("验证")
        toolbar_layout.addWidget(self.validate_btn)
        
        self.format_btn = QPushButton("格式化")
        toolbar_layout.addWidget(self.format_btn)
        
        toolbar_layout.addStretch()
        
        self.status_label = QLabel("就绪")
        toolbar_layout.addWidget(self.status_label)
        
        layout.addLayout(toolbar_layout)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # 文本编辑区域
        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont("Monospace", 10))
        self.text_edit.setPlaceholderText("在此处编辑 JSON 配置文件...")
        
        # 设置语法高亮
        self.highlighter = JsonHighlighter(self.text_edit.document())
        
        # 设置默认文本
        self.text_edit.setPlainText(self.default_profile)
        
        layout.addWidget(self.text_edit)
        
        self.setLayout(layout)
        
        # 设置样式
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
            }
        """)
    
    def init_connections(self):
        """初始化信号连接"""
        self.load_default_btn.clicked.connect(self.load_default_profile)
        self.validate_btn.clicked.connect(self.validate_profile)
        self.format_btn.clicked.connect(self.format_profile)
        
        # 文本变化时发送信号
        self.text_edit.textChanged.connect(self.on_text_changed)
    
    def load_default_profile(self):
        """加载默认配置文件"""
        self.text_edit.setPlainText(self.default_profile)
        self.status_label.setText("已加载默认配置")
        logger.debug("已加载默认配置文件")
    
    def validate_profile(self):
        """验证配置文件"""
        profile_text = self.text_edit.toPlainText().strip()
        
        if not profile_text:
            self.status_label.setText("错误: 配置文件为空")
            self.status_label.setStyleSheet("color: red;")
            return False
        
        try:
            # 尝试解析 JSON
            profile_data = json.loads(profile_text)
            
            # 验证结构
            if self.validate_profile_structure(profile_data):
                self.status_label.setText("验证通过")
                self.status_label.setStyleSheet("color: green;")
                logger.debug("配置文件验证通过")
                return True
            else:
                self.status_label.setText("错误: 配置文件结构无效")
                self.status_label.setStyleSheet("color: red;")
                return False
                
        except json.JSONDecodeError as e:
            self.status_label.setText(f"JSON 解析错误: {e}")
            self.status_label.setStyleSheet("color: red;")
            
            # 高亮错误行
            self.highlight_error_line(e.lineno - 1)
            
            logger.error(f"JSON 解析错误: {e}")
            return False
            
        except Exception as e:
            self.status_label.setText(f"验证错误: {e}")
            self.status_label.setStyleSheet("color: red;")
            logger.error(f"配置文件验证错误: {e}")
            return False
    
    def validate_profile_structure(self, profile_data: Any) -> bool:
        """验证配置文件结构"""
        try:
            from tableturf.manager import Profile
            # 使用 Profile 类的验证方法
            Profile.from_json(json.dumps(profile_data))
            return True
        except:
            # 基本结构验证
            if not isinstance(profile_data, list):
                return False
            
            for item in profile_data:
                if not isinstance(item, dict):
                    return False
                
                required_fields = ["current_level", "current_win", "target_level", "target_win", "deck"]
                for field in required_fields:
                    if field not in item:
                        return False
            
            return True
    
    def format_profile(self):
        """格式化配置文件"""
        profile_text = self.text_edit.toPlainText().strip()
        
        if not profile_text:
            return
        
        try:
            # 解析并重新格式化
            profile_data = json.loads(profile_text)
            formatted_text = json.dumps(profile_data, indent=2, ensure_ascii=False)
            
            self.text_edit.setPlainText(formatted_text)
            self.status_label.setText("已格式化")
            self.status_label.setStyleSheet("color: green;")
            
            logger.debug("配置文件已格式化")
            
        except json.JSONDecodeError as e:
            self.status_label.setText(f"格式化失败: {e}")
            self.status_label.setStyleSheet("color: red;")
            logger.error(f"配置文件格式化失败: {e}")
    
    def highlight_error_line(self, line_number: int):
        """高亮错误行"""
        # 获取文本光标
        cursor = self.text_edit.textCursor()
        
        # 移动到错误行
        cursor.movePosition(QTextCursor.Start)
        for _ in range(line_number):
            cursor.movePosition(QTextCursor.Down)
        
        # 选择整行
        cursor.movePosition(QTextCursor.StartOfLine)
        cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
        
        # 设置高亮格式
        format = QTextCharFormat()
        format.setBackground(QColor(255, 200, 200))  # 浅红色背景
        
        # 应用格式
        cursor.mergeCharFormat(format)
        
        # 滚动到错误行
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
    
    def on_text_changed(self):
        """文本变化处理"""
        self.status_label.setText("已修改")
        self.status_label.setStyleSheet("color: orange;")
        
        # 发送信号
        profile_text = self.text_edit.toPlainText()
        self.profile_changed.emit(profile_text)
    
    def get_profile_json(self) -> Optional[str]:
        """获取配置文件 JSON 字符串"""
        profile_text = self.text_edit.toPlainText().strip()
        
        if not profile_text:
            return None
        
        try:
            # 验证并返回
            if self.validate_profile():
                return profile_text
            else:
                return None
        except:
            return None
    
    def set_profile_json(self, profile_json: str):
        """设置配置文件 JSON 字符串"""
        if profile_json:
            self.text_edit.setPlainText(profile_json)
            self.status_label.setText("已加载")
            self.status_label.setStyleSheet("color: green;")
    
    def cleanup(self):
        """清理资源"""
        logger.debug("配置文件部件资源已清理")