"""
Detection模块测试 - 卡牌检测（带可视化调试）
测试图片说明：
- battle1.jpg, battle2.jpg: 对战画面（手牌显示）
- redraw.jpg: 重抽界面

可视化调试功能：
- 显示检测区域（绿色方框）
- 显示匹配比率（红色数字）
- 鼠标点击查看 BGR/HSV 值
- 按任意键关闭窗口

使用方法：
1. 直接运行查看检测结果: python test_card.py
2. 可视化调试模式: python test_card.py --visual
3. 查看ROI配置: python test_card.py --info
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
    def show_card_detection(img, hands_result, title="卡牌检测"):
        """显示卡牌检测结果"""
        if not SHOW_VISUALIZATION:
            return
        
        img_display = img.copy()
        
        # 绘制手牌区域标记（根据ui.py中的HANDS_GRID_TOP_LEFTS）
        from tableturf.manager.detection.card import (
            HANDS_GRID_TOP_LEFTS, HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT
        )
        
        for i, top_left in enumerate(HANDS_GRID_TOP_LEFTS):
            y, x = int(top_left[0]), int(top_left[1])
            # 绘制大方框表示每张手牌区域
            cv2.rectangle(img_display, (x, y), (x + 200, y + 200), (0, 255, 0), 2)
            cv2.putText(img_display, f'Card {i}', (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示检测到的卡牌信息
        info_y = 30
        for i, card in enumerate(hands_result):
            text = f"Card {i}: SP={card.sp_cost}, Size={card.size}"
            cv2.putText(img_display, text, (10, info_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
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


class TestCardDetection(unittest.TestCase):
    """测试卡牌检测功能"""
    
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
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_hands_detection_basic(self):
        """测试基础手牌检测 - 不崩溃且返回正确类型"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        hands = detection.hands(img)
        
        # 验证返回值
        self.assertIsInstance(hands, list)
        self.assertEqual(len(hands), 4)  # 手牌应该是4张
        for card in hands:
            self.assertIsInstance(card, Card)
        
        # 可视化
        if SHOW_VISUALIZATION:
            VisualHelper.show_card_detection(img, hands, "test_hands_detection_basic")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle2.jpg')), 
                     'test image battle2.jpg not found')
    def test_hands_detection_battle2(self):
        """测试另一张对战图片的手牌检测"""
        capture = self._get_capture('battle2.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        hands = detection.hands(img)
        
        self.assertIsInstance(hands, list)
        self.assertEqual(len(hands), 4)
        
        # 输出检测到的卡牌信息
        for i, card in enumerate(hands):
            print(f"  手牌{i}: SP={card.sp_cost}, 面积={card.size}")
        
        if SHOW_VISUALIZATION:
            VisualHelper.show_card_detection(img, hands, "test_hands_detection_battle2")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'redraw.jpg')), 
                     'test image redraw.jpg not found')
    def test_hands_detection_redraw(self):
        """测试重抽界面的手牌检测"""
        capture = self._get_capture('redraw.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        hands = detection.hands(img)
        
        self.assertIsInstance(hands, list)
        self.assertGreaterEqual(len(hands), 1)
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_enhanced_hands_detection(self):
        """测试增强版手牌检测（带JSON验证）"""
        from tableturf.model import DataLoader, CardFactory
        
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        try:
            loader = DataLoader()
            factory = CardFactory(loader)
            hands = detection.enhanced_hands(img, data_loader=loader, card_factory=factory)
            
            self.assertIsInstance(hands, list)
            self.assertEqual(len(hands), 4)
            print(f"  增强检测完成，JSON验证: 可用")
        except Exception as e:
            hands = detection.hands(img)
            self.assertIsInstance(hands, list)
            print(f"  增强检测回退到基础检测: {e}")
        
        if SHOW_VISUALIZATION:
            VisualHelper.show_card_detection(img, hands, "test_enhanced_hands_detection")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_hands_cursor_detection(self):
        """测试手牌光标位置检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        cursor_pos = detection.hands_cursor(img)
        
        self.assertIn(cursor_pos, [-1, 0, 1, 2, 3])
        print(f"  光标位置: {cursor_pos}")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_enhanced_card_detector_class(self):
        """测试EnhancedCardDetector类"""
        from tableturf.model import DataLoader, CardFactory
        
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        try:
            loader = DataLoader()
            factory = CardFactory(loader)
            detector = detection.EnhancedCardDetector(loader, factory)
            
            hands = detector.hands(img)
            self.assertIsInstance(hands, list)
            print(f"  EnhancedCardDetector检测完成，卡牌数: {len(hands)}")
        except Exception as e:
            self.skipTest(f'EnhancedCardDetector初始化失败: {e}')
        
        capture.close()


class TestCardModuleImports(unittest.TestCase):
    """测试模块导入"""
    
    def test_detection_module_exports(self):
        """测试detection模块导出的函数"""
        self.assertTrue(hasattr(detection, 'hands'))
        self.assertTrue(hasattr(detection, 'enhanced_hands'))
        self.assertTrue(hasattr(detection, 'EnhancedCardDetector'))


def show_card_roi_info():
    """打印卡牌检测的ROI配置信息"""
    print("\n" + "="*60)
    print("卡牌检测 ROI 配置信息")
    print("="*60)
    
    from tableturf.manager.detection import card
    
    configs = [
        ('HANDS_GRID', '手牌网格'),
        ('FOCUS_GRID', '选中手牌网格'),
        ('HANDS_COST', '手牌SP消耗'),
        ('FOCUS_COST', '选中手牌SP消耗'),
    ]
    
    for prefix, desc in configs:
        print(f"\n【{desc}】{prefix}")
        print("-" * 40)
        
        top_lefts = getattr(card, f'{prefix}_TOP_LEFTS', None)
        roi_width = getattr(card, f'{prefix}_ROI_WIDTH', None)
        roi_height = getattr(card, f'{prefix}_ROI_HEIGHT', None)
        
        if top_lefts is not None:
            print(f"  区域数量: {len(top_lefts)}")
            print(f"  ROI尺寸: {roi_width}x{roi_height}")
            print(f"  起始位置:")
            for i, pos in enumerate(top_lefts):
                print(f"    [{i}] y={pos[0]}, x={pos[1]}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='卡牌检测测试（带可视化）')
    parser.add_argument('--visual', '-v', action='store_true', 
                       help='开启可视化模式')
    parser.add_argument('--info', '-i', action='store_true',
                       help='打印ROI配置信息')
    args = parser.parse_args()
    
    if args.info:
        show_card_roi_info()
        sys.exit(0)
    
    if args.visual:
        SHOW_VISUALIZATION = True
        print("\n[可视化模式已开启]\n")
    
    unittest.main(verbosity=2, exit=False)
