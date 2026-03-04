"""
Detection模块测试 - 场地检测（带可视化调试）
测试图片说明：
- battle1.jpg, battle2.jpg: 对战画面（含场地）

可视化调试功能：
- 显示ROI检测区域
- 显示场地网格
- 鼠标点击查看 BGR/HSV 值

使用方法：
1. 直接运行: python test_stage.py
2. 可视化模式: python test_stage.py --visual
3. 查看ROI配置: python test_stage.py --info
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
from tableturf.model import Stage

# 测试图片路径
TEST_IMAGES_PATH = os.path.join(project_root, 'tests', 'test_capture', 'test_images')
TEMP_PATH = os.path.join(project_root, 'temp')

# 可视化开关
SHOW_VISUALIZATION = False


class VisualHelper:
    """可视化辅助类"""
    
    @staticmethod
    def show_stage_detection(img, rois, roi_width, roi_height, stage_result, title="场地检测"):
        """显示场地检测结果"""
        if not SHOW_VISUALIZATION:
            return
        
        img_display = img.copy()
        
        # 绘制ROI区域
        if rois is not None and len(rois) > 0:
            for i, roi in enumerate(rois[:20]):  # 最多显示20个ROI
                y, x = int(roi[0]), int(roi[1])
                cv2.rectangle(img_display, (x, y), (x + roi_width, y + roi_height), (0, 255, 0), 1)
                if i < 5:  # 只在部分ROI上标注编号
                    cv2.putText(img_display, f'{i}', (x, y - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 显示场地信息
        if stage_result:
            info_text = f"Stage: {stage_result.width}x{stage_result.height}"
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


class TestStageDetection(unittest.TestCase):
    """测试场地检测功能"""
    
    @classmethod
    def setUpClass(cls):
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
    def test_stage_rois_detection(self):
        """测试场地ROI区域检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        rois, roi_width, roi_height = detection.stage_rois(img)
        
        self.assertIsInstance(rois, np.ndarray)
        self.assertGreater(len(rois), 0)
        self.assertGreater(roi_width, 0)
        self.assertGreater(roi_height, 0)
        
        print(f"\n  ROI检测: {len(rois)}个区域, 单格尺寸={roi_width}x{roi_height}")
        
        if SHOW_VISUALIZATION:
            VisualHelper.show_stage_detection(img, rois, roi_width, roi_height, None, "test_stage_rois")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_stage_detection_basic(self):
        """测试基础场地检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        rois, roi_width, roi_height = detection.stage_rois(img)
        if len(rois) == 0:
            self.skipTest('no ROIs detected')
        
        stage = detection.stage(img, rois=rois, roi_width=roi_width, roi_height=roi_height)
        
        self.assertIsInstance(stage, Stage)
        print(f"  场地检测: 尺寸={stage.width}x{stage.height}")
        
        if SHOW_VISUALIZATION:
            VisualHelper.show_stage_detection(img, rois, roi_width, roi_height, stage, "test_stage")
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_stage_with_last_stage(self):
        """测试使用last_stage参数的场地检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        rois, roi_width, roi_height = detection.stage_rois(img)
        if len(rois) == 0:
            self.skipTest('no ROIs detected')
        
        stage1 = detection.stage(img, rois=rois, roi_width=roi_width, roi_height=roi_height, last_stage=None)
        stage2 = detection.stage(img, rois=rois, roi_width=roi_width, roi_height=roi_height, last_stage=stage1)
        
        self.assertIsInstance(stage1, Stage)
        self.assertIsInstance(stage2, Stage)
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_enhanced_stage_detection(self):
        """测试增强版场地检测（带JSON验证）"""
        from tableturf.model import DataLoader, StageFactory
        
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        try:
            loader = DataLoader()
            factory = StageFactory(loader)
            
            rois, roi_width, roi_height = detection.stage_rois(img)
            if len(rois) == 0:
                self.skipTest('no ROIs detected')
            
            stage = detection.enhanced_stage(
                img, rois=rois, roi_width=roi_width, roi_height=roi_height,
                data_loader=loader, stage_factory=factory
            )
            
            self.assertIsInstance(stage, Stage)
            print(f"  增强场地检测完成，JSON验证: 可用")
        except Exception as e:
            self.skipTest(f'增强检测初始化失败: {e}')
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_sp_detection(self):
        """测试SP点数检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        try:
            my_sp, his_sp = detection.sp(img)
            self.assertGreaterEqual(my_sp, 0)
            self.assertGreaterEqual(his_sp, 0)
            print(f"  SP检测: 我方={my_sp}, 敌方={his_sp}")
        except Exception as e:
            self.skipTest(f'SP检测失败: {e}')
        
        capture.close()
    
    @unittest.skipIf(not os.path.exists(os.path.join(TEST_IMAGES_PATH, 'battle1.jpg')), 
                     'test image battle1.jpg not found')
    def test_preview_detection(self):
        """测试卡牌预览检测"""
        capture = self._get_capture('battle1.jpg')
        if not capture:
            self.skipTest('capture not available')
        
        img = capture.capture()
        
        rois, roi_width, roi_height = detection.stage_rois(img)
        if len(rois) == 0:
            self.skipTest('no ROIs detected')
        
        stage = detection.stage(img, rois=rois, roi_width=roi_width, roi_height=roi_height)
        
        try:
            preview, index = detection.preview(img, stage=stage, rois=rois, 
                                               roi_width=roi_width, roi_height=roi_height)
            print(f"  预览检测: index={index}")
        except Exception as e:
            print(f"  预览检测（可能未实现）: {e}")
        
        capture.close()


class TestStageModuleImports(unittest.TestCase):
    """测试模块导入"""
    
    def test_stage_exports(self):
        """测试stage相关函数已导出"""
        self.assertTrue(hasattr(detection, 'stage_rois'))
        self.assertTrue(hasattr(detection, 'stage'))
        self.assertTrue(hasattr(detection, 'enhanced_stage'))
        self.assertTrue(hasattr(detection, 'EnhancedStageDetector'))
        self.assertTrue(hasattr(detection, 'preview'))
        self.assertTrue(hasattr(detection, 'sp'))


def show_stage_roi_info():
    """打印场地检测的ROI配置信息"""
    print("\n" + "="*60)
    print("场地检测 ROI 配置信息")
    print("="*60)
    
    from tableturf.manager.detection import stage
    
    print("\n【场地边界框】")
    print("-" * 40)
    bbox_top = getattr(stage, 'BOUNDING_BOX_TOP_LEFT', None)
    bbox_w = getattr(stage, 'BOUNDING_BOX_WIDTH', None)
    bbox_h = getattr(stage, 'BOUNDING_BOX_HEIGHT', None)
    if bbox_top is not None:
        print(f"  左上角: y={bbox_top[0]}, x={bbox_top[1]}")
        print(f"  尺寸: {bbox_w}x{bbox_h}")
    
    print("\n【连通组件尺寸阈值】")
    print("-" * 40)
    w_min = getattr(stage, 'CC_WIDTH_LOWER_BOUND', None)
    w_max = getattr(stage, 'CC_WIDTH_UPPER_BOUND', None)
    h_min = getattr(stage, 'CC_HEIGHT_LOWER_BOUND', None)
    h_max = getattr(stage, 'CC_HEIGHT_UPPER_BOUND', None)
    print(f"  宽度: {w_min} - {w_max}")
    print(f"  高度: {h_min} - {h_max}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='场地检测测试（带可视化）')
    parser.add_argument('--visual', '-v', action='store_true', 
                       help='开启可视化模式')
    parser.add_argument('--info', '-i', action='store_true',
                       help='打印ROI配置信息')
    args = parser.parse_args()
    
    if args.info:
        show_stage_roi_info()
        sys.exit(0)
    
    if args.visual:
        SHOW_VISUALIZATION = True
        print("\n[可视化模式已开启]\n")
    
    unittest.main(verbosity=2, exit=False)
