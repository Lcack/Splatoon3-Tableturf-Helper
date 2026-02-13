import io
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt

from tableturf.manager.detection.debugger.interface import Debugger


class QtDebugger(Debugger):
    def __init__(self):
        # 初始化空图像
        __empty = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.__empty_image = self.__convert_to_qimage(__empty)
        # 存储图像的字典
        self.__images = dict()
        # 图像更新信号的回调函数
        self.__update_callback = None

    def show(self, name: str, img: np.ndarray):
        """
        显示图像，将 NumPy 数组转换为 QImage 并存储
        """
        q_image = self.__convert_to_qimage(img)
        self.__images[name] = q_image
        # 如果设置了回调，通知更新
        if self.__update_callback:
            self.__update_callback(name)

    def get(self, name: str) -> QImage:
        """
        获取指定名称的图像
        """
        return self.__images.get(name, self.__empty_image)

    def list(self):
        """
        返回所有已存储的图像名称
        """
        return self.__images.keys()

    def set_update_callback(self, callback):
        """
        设置图像更新时的回调函数
        """
        self.__update_callback = callback

    def __convert_to_qimage(self, img: np.ndarray) -> QImage:
        """
        将 NumPy 数组转换为 QImage
        """
        # 确保图像是 RGB 格式
        if len(img.shape) == 3 and img.shape[2] == 3:
            # OpenCV 图像是 BGR 格式，需要转换为 RGB
            img_rgb = img[:, :, ::-1].copy()
            height, width, channel = img_rgb.shape
            bytes_per_line = 3 * width
            return QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif len(img.shape) == 2:
            # 灰度图像
            height, width = img.shape
            return QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        else:
            # 其他格式，返回空图像
            return self.__empty_image


# 创建全局实例
qt_debugger = QtDebugger()