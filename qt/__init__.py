"""
Qt 桌面应用模块
"""

from qt.main import main as run_app
from qt.home import MainWindow, ControllerWidget, VideoWidget, ProfileWidget, ControlWidget
from qt.debug import DebugWindow

__all__ = [
    'run_app',
    'MainWindow',
    'ControllerWidget',
    'VideoWidget',
    'ProfileWidget',
    'ControlWidget',
    'DebugWindow',
]

__version__ = "1.0.0"