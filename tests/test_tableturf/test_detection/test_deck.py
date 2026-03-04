"""
Detection模块测试 - 卡组检测（带可视化调试）
测试图片说明：
- chooseDeck.jpg: 卡组选择界面

可视化调试功能：
- 显示检测区域（绿色方框）
- 显示卡牌编号和SP消耗
- 鼠标点击查看 BGR/HSV 值
- 按任意键关闭窗口

使用方法：
1. 直接运行: python test_deck.py
2. 可视化模式: python test_deck.py --visual
3. 查看ROI配置: python test_deck.py --info
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
from tableturf.model import Card

# 测试图片路径
TEST_IMAGES_PATH = os.path.join(project_root, 'tests', 'test_capture', 'test_images')
TEMP_PATH = os.path.join(project_root, 'temp')

# 可视化开关
SHOW_VISUALIZATION = False


class VisualHelper:
    """可视化辅助类"""
    
    @staticmethod
    def show_deck_detection(img, deck_result, title="卡组检测"):
        """显示卡组检测结果"""
        if not SHOW_VISUALIZATION:
            return
        
        img_display = img.copy()
        
        # 绘制卡组区域标记
        from tableturf.manager.detection.deck import (
            DECK_GRID_TOP_LEFTS, DECK_GRID_ROI_WIDTH, DECK_GRID_ROI_HEIGHT
        )
        
        for i, top_left in enumerate(DECK_GRID_TOP_LEFTS):
            y, x = int(top_left[0]), int(top_left[1])
            cv2.rectangle(img_display, (x, y), (x + 100, y + 100), (0, 255, 0), 2)
            if i < len(deck_result):
                text = f"{deck_result[i].sp_cost}"
                cv2.putText(img_display, text, (x + 30, y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 显示卡组信息
        info_text = f"Cards: {len(deck_result)}"
        cv2.putText(img_display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 鼠标回调
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bgr = img[y, x]
                hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                print(f"    [点击] 位置=({x}, {y}), BGR={bgr}, HSV={hsv}")
        
        cv2.imshow(title, img_display)
        cv2.setMouseCallback(title, on_mouse)
        print(f"    按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyWindow(title)


class TestDeckDetection(unittest.TestCase):
    """测试卡组检测功能"""
    
    @classmethod
    def setUpClass(cls):
        cls.has_test_images = os.path.exists(TEST_IMAGES_PATH)
        cls.has_temp = os.path.exists(TEMP_PATH)
    
    def _get_capture(self, image_name=None, path=None):
        if path and os.path.exists(path):
            return FileLoader(path=path)
        if image_name and self.has_test_images:
            image_path = os.path.join(TEST_IMAGES_PATH, image_name)
            if os.path.exists(image_path):
                return FileLoader(files=[image_path])
        return None
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'chooseDeck.jpg')), 
                     'test image chooseDeck.jpg not found')
    def test_deck_detection_basic(self):
        """测试基础卡组检测"""
        capture = self._get_capture('chooseDeck.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        deck = detection.deck(img)
        
        self.assertIsInstance(deck, list)
        self.assertEqual(len(deck), 15)
        for card in deck:
            self.assertIsInstance(card, Card)
        
        print(f"\n  卡组检测完成，共{len(deck)}张卡牌")
        for i, card in enumerate(deck[:5]):
            print(f"    卡牌{i}: SP={card.sp_cost}, 面积={card.size}")
        
        if SHOW_VISUALIZATION:
            VisualHelper.show_deck_detection(img, deck, "test_deck_basic")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'chooseDeck.jpg')), 
                     'test image chooseDeck.jpg not found')
    def test_deck_cursor_detection(self):
        """测试卡组光标位置检测"""
        capture = self._get_capture('chooseDeck.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        cursor_pos = detection.deck_cursor(img)
        
        self.assertIn(cursor_pos, list(range(-1, 15)))
        print(f"  卡组光标位置: {cursor_pos}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'chooseDeck.jpg')), 
                     'test image chooseDeck.jpg not found')
    def test_deck_card_properties(self):
        """测试检测到的卡牌属性合理性"""
        capture = self._get_capture('chooseDeck.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        deck = detection.deck(img)
        
        for i, card in enumerate(deck):
            self.assertGreaterEqual(card.sp_cost, 0)
            self.assertLessEqual(card.sp_cost, 10)
            self.assertGreater(card.size, 0)
            self.assertLessEqual(card.size, 16)
        
        print(f"  所有{len(deck)}张卡牌属性检查通过")
        
        capture.close()


class TestDeckModuleImports(unittest.TestCase):
    """测试模块导入"""
    
    def test_deck_exported(self):
        self.assertTrue(hasattr(detection, 'deck'))
        self.assertTrue(hasattr(detection, 'deck_cursor'))


def show_deck_roi_info():
    """打印卡组检测的ROI配置信息"""
    print("\n" + "="*60)
    print("卡组检测 ROI 配置信息")
    print("="*60)
    
    from tableturf.manager.detection import deck
    
    print("\n【卡组网格】DECK_GRID")
    print("-" * 40)
    top_lefts = getattr(deck, 'DECK_GRID_TOP_LEFTS', None)
    roi_w = getattr(deck, 'DECK_GRID_ROI_WIDTH', None)
    roi_h = getattr(deck, 'DECK_GRID_ROI_HEIGHT', None)
    
    if top_lefts is not None:
        print(f"  卡牌数量: {len(top_lefts)}")
        print(f"  网格ROI尺寸: {roi_w}x{roi_h}")
        print(f"  位置 (前5个):")
        for i, pos in enumerate(top_lefts[:5]):
            print(f"    [{i}] y={pos[0]}, x={pos[1]}")
    
    print("\n【SP消耗】DECK_COST")
    print("-" * 40)
    cost_tops = getattr(deck, 'DECK_COST_TOP_LEFTS', None)
    cost_w = getattr(deck, 'DECK_COST_ROI_WIDTH', None)
    cost_h = getattr(deck, 'DECK_COST_ROI_HEIGHT', None)
    
    if cost_tops is not None:
        print(f"  数量: {len(cost_tops)}")
        print(f"  ROI尺寸: {cost_w}x{cost_h}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='卡组检测测试（带可视化）')
    parser.add_argument('--visual', '-v', action='store_true', 
                       help='开启可视化模式')
    parser.add_argument('--info', '-i', action='store_true',
                       help='打印ROI配置信息')
    args = parser.parse_args()
    
    if args.info:
        show_deck_roi_info()
        sys.exit(0)
    
    if args.visual:
        SHOW_VISUALIZATION = True
        print("\n[可视化模式已开启]\n")
    
    unittest.main(verbosity=2, exit=False)
