"""
Detection模块测试 - UI检测（带可视化调试）
测试图片说明：
- chooseDeck.jpg: 卡组选择（deck_cursor）
- battle1.jpg, battle2.jpg: 对战画面（hands_cursor）
- redraw.jpg: 重抽界面（redraw_cursor）
- greeting.jpg: 打招呼界面

可视化调试功能：
- 显示检测区域（绿色方框）
- 显示匹配比率（红色数字）
- 鼠标点击查看 BGR/HSV 值
- 按任意键关闭窗口

使用方法：
1. 直接运行查看检测结果: python test_ui.py
2. 可视化调试模式: 设置 SHOW_VISUALIZATION = True
"""

import os
import unittest
import sys
import cv2
import numpy as np

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, project_root)

from capture import FileLoader
from tableturf.manager.detection.debugger import OpenCVDebugger
from tableturf.manager import detection
from tableturf.manager.data import Result

# 测试图片路径
TEST_IMAGES_PATH = os.path.join(project_root, 'tests', 'test_capture', 'test_images')
TEMP_PATH = os.path.join(project_root, 'temp')

# 可视化开关 - 设为True会显示检测窗口
SHOW_VISUALIZATION = False


class VisualDebugger(OpenCVDebugger):
    """
    可视化调试器 - 显示检测ROI和匹配比率
    
    功能：
    1. 绿色方框：标记检测区域
    2. 绿色数字：ROI编号
    3. 红色数字：匹配比率（0-1之间）
    4. 鼠标点击：打印该位置BGR/HSV值
    """
    
    def show_with_info(self, name: str, img: np.ndarray, rois=None, ratios=None, 
                       width=None, height=None, wait_key=True):
        """
        显示图像并标注ROI信息
        
        Args:
            name: 窗口名称
            img: 原始图像
            rois: ROI左上角坐标列表 [(y,x), ...]
            ratios: 每个ROI的匹配比率
            width: ROI宽度
            height: ROI高度
            wait_key: 是否等待按键
        """
        img_display = img.copy()
        
        # 转换numpy坐标到opencv坐标
        if rois is not None and width is not None and height is not None:
            for i, roi in enumerate(rois):
                # roi是[y, x]格式，需要转为[x, y]给cv2
                x, y = int(roi[1]), int(roi[0])
                
                # 画绿色方框
                cv2.rectangle(img_display, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # 显示ROI编号（绿色）
                cv2.putText(img_display, f'{i}', (x + 2, y + height - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示匹配比率（红色）
                if ratios is not None and i < len(ratios):
                    ratio_text = f'{ratios[i]:.2f}'
                    cv2.putText(img_display, ratio_text, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 鼠标回调函数 - 打印像素值
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bgr = img[y, x]
                hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                print(f"    [点击] 位置=({x}, {y}), BGR={bgr}, HSV={hsv}")
        
        cv2.imshow(name, img_display)
        cv2.setMouseCallback(name, on_mouse_click)
        
        if wait_key:
            print(f"    按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyWindow(name)
        else:
            cv2.waitKey(500)  # 显示500ms
            cv2.destroyWindow(name)


def visualize_cursor_detection(img, cursor_func, func_name, rois, width, height, 
                               hsv_lower, hsv_upper, threshold):
    """
    可视化光标检测过程
    
    Args:
        img: 输入图像
        cursor_func: 检测函数
        func_name: 函数名称（用于显示）
        rois: ROI坐标列表
        width, height: ROI尺寸
        hsv_lower, hsv_upper: HSV颜色范围
        threshold: 匹配阈值
    """
    if not SHOW_VISUALIZATION:
        return cursor_func(img)
    
    # 计算每个ROI的匹配比率
    def calc_ratio(top_left):
        roi = img[int(top_left[0]):int(top_left[0]) + height, 
                  int(top_left[1]):int(top_left[1]) + width]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        return np.sum(mask == 255) / (width * height)
    
    ratios = [calc_ratio(roi) for roi in rois]
    
    # 执行检测
    result = cursor_func(img)
    
    # 显示可视化结果
    print(f"\n  [{func_name}] 检测结果: {result}")
    print(f"  ROI数量: {len(rois)}, 阈值: {threshold}")
    for i, (roi, ratio) in enumerate(zip(rois, ratios)):
        status = "✓" if ratio >= threshold else "✗"
        print(f"    ROI{i}: 比率={ratio:.3f} {status}")
    
    debugger = VisualDebugger()
    debugger.show_with_info(f'{func_name} - 点击图像查看像素值', img, rois, ratios, width, height)
    
    return result


class TestUICursorDetection(unittest.TestCase):
    """测试UI光标检测功能"""
    
    @classmethod
    def setUpClass(cls):
        """检查测试图片是否存在"""
        cls.has_test_images = os.path.exists(TEST_IMAGES_PATH)
        cls.has_temp = os.path.exists(TEMP_PATH)
    
    def _get_capture(self, image_name=None, path=None):
        """辅助方法：获取图像捕获器"""
        if path and os.path.exists(path):
            return FileLoader(path=path)
        if image_name and self.has_test_images:
            image_path = os.path.join(TEST_IMAGES_PATH, image_name)
            if os.path.exists(image_path):
                return FileLoader(files=[image_path])
        return None
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'chooseDeck.jpg')), 
                     'test image chooseDeck.jpg not found')
    def test_deck_cursor(self):
        """测试卡组光标检测"""
        capture = self._get_capture('chooseDeck.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        if SHOW_VISUALIZATION:
            # 从ui.py导入常量
            from tableturf.manager.detection.ui import (
                DECK_CURSOR_NUMPY_ROI_TOP_LEFTS, DECK_CURSOR_ROI_WIDTH, 
                DECK_CURSOR_ROI_HEIGHT, DECK_CURSOR_COLOR_HSV_LOWER_BOUND,
                DECK_CURSOR_COLOR_HSV_UPPER_BOUND, DECK_CURSOR_PIXEL_RATIO
            )
            cursor_pos = visualize_cursor_detection(
                img, detection.deck_cursor, 'deck_cursor',
                DECK_CURSOR_NUMPY_ROI_TOP_LEFTS, DECK_CURSOR_ROI_WIDTH, 
                DECK_CURSOR_ROI_HEIGHT, DECK_CURSOR_COLOR_HSV_LOWER_BOUND,
                DECK_CURSOR_COLOR_HSV_UPPER_BOUND, DECK_CURSOR_PIXEL_RATIO
            )
        else:
            cursor_pos = detection.deck_cursor(img)
        
        # 卡组光标范围: -1(未找到) 到 14(15个卡组)
        self.assertIn(cursor_pos, list(range(-1, 15)))
        print(f"  卡组光标: {cursor_pos}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_hands_cursor(self):
        """测试手牌光标检测（带可视化）"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        if SHOW_VISUALIZATION:
            from tableturf.manager.detection.ui import (
                HANDS_CURSOR_NUMPY_ROI_TOP_LEFTS, HANDS_CURSOR_ROI_WIDTH,
                HANDS_CURSOR_ROI_HEIGHT, HANDS_CURSOR_COLOR_HSV_LOWER_BOUND,
                HANDS_CURSOR_COLOR_HSV_UPPER_BOUND, HANDS_CURSOR_PIXEL_RATIO
            )
            cursor_pos = visualize_cursor_detection(
                img, detection.hands_cursor, 'hands_cursor',
                HANDS_CURSOR_NUMPY_ROI_TOP_LEFTS, HANDS_CURSOR_ROI_WIDTH,
                HANDS_CURSOR_ROI_HEIGHT, HANDS_CURSOR_COLOR_HSV_LOWER_BOUND,
                HANDS_CURSOR_COLOR_HSV_UPPER_BOUND, HANDS_CURSOR_PIXEL_RATIO
            )
        else:
            cursor_pos = detection.hands_cursor(img)
        
        # 手牌光标范围: -1(未找到) 到 5(4张手牌+Skip+Special)
        self.assertIn(cursor_pos, list(range(-1, 6)))
        print(f"  手牌光标: {cursor_pos}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'redraw.jpg')), 
                     'test image redraw.jpg not found')
    def test_redraw_cursor(self):
        """测试重抽光标检测"""
        capture = self._get_capture('redraw.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        if SHOW_VISUALIZATION:
            from tableturf.manager.detection.ui import (
                REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS, REDRAW_CURSOR_ROI_WIDTH,
                REDRAW_CURSOR_ROI_HEIGHT, REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND,
                REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND, REDRAW_CURSOR_PIXEL_RATIO
            )
            cursor_pos = visualize_cursor_detection(
                img, detection.redraw_cursor, 'redraw_cursor',
                REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS, REDRAW_CURSOR_ROI_WIDTH,
                REDRAW_CURSOR_ROI_HEIGHT, REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND,
                REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND, REDRAW_CURSOR_PIXEL_RATIO
            )
        else:
            cursor_pos = detection.redraw_cursor(img)
        
        self.assertIn(cursor_pos, [-1, 0, 1])
        print(f"  重抽光标: {cursor_pos}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_special_on(self):
        """测试特殊攻击状态检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        is_special = detection.special_on(img)
        
        self.assertIsInstance(is_special, bool)
        print(f"  特殊攻击状态: {is_special}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_skip(self):
        """测试跳过按钮检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        is_skip = detection.skip(img)
        
        self.assertIsInstance(is_skip, bool)
        print(f"  跳过按钮: {is_skip}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'greeting.jpg')), 
                     'test image greeting.jpg not found')
    def test_replay_cursor(self):
        """测试回放光标检测"""
        capture = self._get_capture('greeting.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        if SHOW_VISUALIZATION:
            from tableturf.manager.detection.ui import (
                REPLAY_CURSOR_NUMPY_ROI_TOP_LEFTS, REPLAY_CURSOR_ROI_WIDTH,
                REPLAY_CURSOR_ROI_HEIGHT, REPLAY_CURSOR_COLOR_HSV_LOWER_BOUND,
                REPLAY_CURSOR_COLOR_HSV_UPPER_BOUND, REPLAY_CURSOR_PIXEL_RATIO
            )
            cursor_pos = visualize_cursor_detection(
                img, detection.replay_cursor, 'replay_cursor',
                REPLAY_CURSOR_NUMPY_ROI_TOP_LEFTS, REPLAY_CURSOR_ROI_WIDTH,
                REPLAY_CURSOR_ROI_HEIGHT, REPLAY_CURSOR_COLOR_HSV_LOWER_BOUND,
                REPLAY_CURSOR_COLOR_HSV_UPPER_BOUND, REPLAY_CURSOR_PIXEL_RATIO
            )
        else:
            cursor_pos = detection.replay_cursor(img)
        
        self.assertIn(cursor_pos, [-1, 0, 1])
        print(f"  回放光标: {cursor_pos}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'greeting.jpg')), 
                     'test image greeting.jpg not found')
    def test_giveup_cursor(self):
        """测试放弃光标检测"""
        capture = self._get_capture('greeting.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        if SHOW_VISUALIZATION:
            from tableturf.manager.detection.ui import (
                GIVEUP_CURSOR_NUMPY_ROI_TOP_LEFTS, GIVEUP_CURSOR_ROI_WIDTH,
                GIVEUP_CURSOR_ROI_HEIGHT, GIVEUP_CURSOR_COLOR_HSV_LOWER_BOUND,
                GIVEUP_CURSOR_COLOR_HSV_UPPER_BOUND, GIVEUP_CURSOR_PIXEL_RATIO
            )
            cursor_pos = visualize_cursor_detection(
                img, detection.giveup_cursor, 'giveup_cursor',
                GIVEUP_CURSOR_NUMPY_ROI_TOP_LEFTS, GIVEUP_CURSOR_ROI_WIDTH,
                GIVEUP_CURSOR_ROI_HEIGHT, GIVEUP_CURSOR_COLOR_HSV_LOWER_BOUND,
                GIVEUP_CURSOR_COLOR_HSV_UPPER_BOUND, GIVEUP_CURSOR_PIXEL_RATIO
            )
        else:
            cursor_pos = detection.giveup_cursor(img)
        
        self.assertIn(cursor_pos, [-1, 0, 1])
        print(f"  放弃光标: {cursor_pos}")
        
        capture.close()


class TestUIStateDetection(unittest.TestCase):
    """测试UI状态检测功能"""
    
    @classmethod
    def setUpClass(cls):
        """检查测试图片是否存在"""
        cls.has_test_images = os.path.exists(TEST_IMAGES_PATH)
    
    def _get_capture(self, image_name=None):
        """辅助方法：获取图像捕获器"""
        if image_name and self.has_test_images:
            image_path = os.path.join(TEST_IMAGES_PATH, image_name)
            if os.path.exists(image_path):
                return FileLoader(files=[image_path])
        return None
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_result_detection(self):
        """测试对战结果检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        result = detection.result(img)
        
        self.assertIsInstance(result, Result)
        self.assertIn(result, [Result.Win, Result.Loss, Result.Draw])
        print(f"\n  对战结果: {result}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'chooseDifficulty.jpg')), 
                     'test image chooseDifficulty.jpg not found')
    def test_level_detection(self):
        """测试难度等级检测"""
        capture = self._get_capture('chooseDifficulty.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        is_level_screen = detection.level(img)
        
        self.assertIsInstance(is_level_screen, bool)
        print(f"  难度选择界面: {is_level_screen}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'chooseNPC.jpg')), 
                     'test image chooseNPC.jpg not found')
    def test_start_detection(self):
        """测试开始按钮检测"""
        capture = self._get_capture('chooseNPC.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        is_start = detection.start(img)
        
        self.assertIsInstance(is_start, bool)
        print(f"  开始按钮: {is_start}")
        
        capture.close()


class TestUIModuleImports(unittest.TestCase):
    """测试模块导入"""
    
    def test_ui_exports(self):
        """测试UI相关函数已导出"""
        self.assertTrue(hasattr(detection, 'deck_cursor'))
        self.assertTrue(hasattr(detection, 'hands_cursor'))
        self.assertTrue(hasattr(detection, 'redraw_cursor'))
        self.assertTrue(hasattr(detection, 'replay_cursor'))
        self.assertTrue(hasattr(detection, 'giveup_cursor'))
        self.assertTrue(hasattr(detection, 'special_on'))
        self.assertTrue(hasattr(detection, 'skip'))
        self.assertTrue(hasattr(detection, 'result'))
        self.assertTrue(hasattr(detection, 'level'))
        self.assertTrue(hasattr(detection, 'start'))


class TestVisualizationTools(unittest.TestCase):
    """测试可视化工具"""
    
    @unittest.skipIf(not SHOW_VISUALIZATION, 'visualization disabled')
    def test_visual_debugger(self):
        """测试可视化调试器"""
        # 创建一个简单测试图像
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[:] = (100, 100, 100)  # 灰色背景
        
        # 定义一些测试ROI
        rois = np.array([[50, 50], [50, 150], [120, 50]])
        ratios = [0.85, 0.12, 0.67]
        
        debugger = VisualDebugger()
        debugger.show_with_info('Test Visualizer', img, rois, ratios, 40, 40)
        
        print("  可视化调试器测试完成")


def show_detection_roi_info():
    """
    打印所有光标检测的ROI配置信息
    用于参考调整参数
    """
    print("\n" + "="*60)
    print("Detection模块 ROI 配置信息")
    print("="*60)
    
    from tableturf.manager.detection import ui
    
    configs = [
        ('hands_cursor', '手牌光标', 'HANDS_CURSOR'),
        ('deck_cursor', '卡组光标', 'DECK_CURSOR'),
        ('redraw_cursor', '重抽光标', 'REDRAW_CURSOR'),
        ('replay_cursor', '回放光标', 'REPLAY_CURSOR'),
        ('giveup_cursor', '放弃光标', 'GIVEUP_CURSOR'),
    ]
    
    for func_name, desc, prefix in configs:
        print(f"\n【{desc}】{func_name}")
        print("-" * 40)
        
        rois = getattr(ui, f'{prefix}_NUMPY_ROI_TOP_LEFTS', None)
        width = getattr(ui, f'{prefix}_ROI_WIDTH', None)
        height = getattr(ui, f'{prefix}_ROI_HEIGHT', None)
        threshold = getattr(ui, f'{prefix}_PIXEL_RATIO', None)
        
        if rois is not None:
            print(f"  ROI数量: {len(rois)}")
            print(f"  ROI尺寸: {width}x{height}")
            print(f"  匹配阈值: {threshold}")
            print(f"  检测位置:")
            for i, roi in enumerate(rois):
                print(f"    [{i}] y={roi[0]}, x={roi[1]}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='UI检测测试（带可视化）')
    parser.add_argument('--visual', '-v', action='store_true', 
                       help='开启可视化模式，显示检测窗口')
    parser.add_argument('--info', '-i', action='store_true',
                       help='打印ROI配置信息，不运行测试')
    args = parser.parse_args()
    
    if args.info:
        show_detection_roi_info()
        sys.exit(0)
    
    if args.visual:
        SHOW_VISUALIZATION = True
        print("\n[可视化模式已开启]")
        print("测试时会弹出窗口显示检测区域")
        print("绿色方框=ROI区域, 红色数字=匹配比率")
        print("点击图像可查看像素BGR/HSV值\n")
    
    unittest.main(verbosity=2, exit=False)
