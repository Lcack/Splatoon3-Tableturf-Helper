from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit


class ProfileEditor(QWidget):
    """
    Profile配置编辑器
    """
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 配置文本框
        self.layout.addWidget(QLabel("Profile:"))
        self.profile_edit = QTextEdit()
        self.profile_edit.setFixedSize(300, 720)
        # 设置默认配置
        default_profile = '''[
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
]'''
        self.profile_edit.setText(default_profile)
        self.layout.addWidget(self.profile_edit)
    
    def get_profile_text(self):
        """
        获取当前的Profile配置文本
        """
        return self.profile_edit.toPlainText()
    
    def set_profile_text(self, text):
        """
        设置Profile配置文本
        """
        self.profile_edit.setText(text)