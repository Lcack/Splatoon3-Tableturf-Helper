from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                               QLabel, QPushButton, QSplitter, QScrollArea)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer

from .debugger import qt_debugger


class DebugWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Debug Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置布局
        main_layout = QVBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：图像列表
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        
        list_layout.addWidget(QLabel("Debug Images:"))
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        list_layout.addWidget(self.image_list)
        
        # 刷新按钮
        refresh_button = QPushButton("Refresh List")
        refresh_button.clicked.connect(self.update_image_list)
        list_layout.addWidget(refresh_button)
        
        splitter.addWidget(list_widget)
        splitter.setSizes([300, 1100])  # 设置初始大小
        
        # 右侧：图像显示
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        
        # 添加滚动区域，以便查看大图像
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        
        display_layout.addWidget(scroll_area)
        splitter.addWidget(display_widget)
        
        main_layout.addWidget(splitter)
        
        # 初始化
        self.update_image_list()
        
        # 设置图像更新回调
        qt_debugger.set_update_callback(self.on_image_updated)
        
        # 设置定时器，定期更新图像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_current_image)
        self.timer.start(500)  # 每 500 毫秒更新一次
        
        # 当前选中的图像名称
        self.current_image_name = None
    
    def update_image_list(self):
        """
        更新图像列表
        """
        # 保存当前选中的项
        selected_items = self.image_list.selectedItems()
        selected_name = selected_items[0].text() if selected_items else None
        
        # 清空列表并重新添加
        self.image_list.clear()
        for name in qt_debugger.list():
            self.image_list.addItem(name)
        
        # 重新选择之前的项
        if selected_name:
            for i in range(self.image_list.count()):
                if self.image_list.item(i).text() == selected_name:
                    self.image_list.setCurrentItem(self.image_list.item(i))
                    self.on_image_selected(self.image_list.item(i))
                    break
    
    def on_image_selected(self, item):
        """
        当选中图像时的处理
        """
        self.current_image_name = item.text()
        self.update_current_image()
    
    def update_current_image(self):
        """
        更新当前显示的图像
        """
        if self.current_image_name:
            q_image = qt_debugger.get(self.current_image_name)
            pixmap = QPixmap.fromImage(q_image)
            # 调整图像大小以适应显示区域
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def on_image_updated(self, name):
        """
        当图像更新时的处理
        """
        # 检查图像是否在列表中
        found = False
        for i in range(self.image_list.count()):
            if self.image_list.item(i).text() == name:
                found = True
                break
        
        # 如果不在列表中，更新列表
        if not found:
            self.update_image_list()
        
        # 如果是当前选中的图像，更新显示
        if name == self.current_image_name:
            self.update_current_image()
    
    def resizeEvent(self, event):
        """
        窗口大小改变时的处理
        """
        super().resizeEvent(event)
        self.update_current_image()