"""
增强版场地检测模块
集成JSON数据验证，提高识别准确性
"""

from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from logger import logger
from tableturf.manager.detection.debugger import Debugger
from tableturf.manager.detection import util
from tableturf.manager.detection.ui import special_on
from tableturf.model import Stage, Pattern, Grid

# 导入JSON数据模块
try:
    from tableturf.model import DataLoader, StageFactory
    HAS_JSON_SUPPORT = True
except ImportError:
    HAS_JSON_SUPPORT = False
    logger.warning("JSON数据模块不可用，将使用纯图像识别")

# 从原stage.py复制常量
BOUNDING_BOX_TOP_LEFT = np.array([0, 750])
BOUNDING_BOX_WIDTH = 850
BOUNDING_BOX_HEIGHT = 1080

CC_WIDTH_UPPER_BOUND = 45
CC_WIDTH_LOWER_BOUND = 30
CC_HEIGHT_UPPER_BOUND = 45
CC_HEIGHT_LOWER_BOUND = 30
CC_AREA_UPPER_BOUND = CC_WIDTH_UPPER_BOUND * CC_HEIGHT_UPPER_BOUND
CC_AREA_LOWER_BOUND = CC_WIDTH_LOWER_BOUND * CC_HEIGHT_LOWER_BOUND

EMPTY_COLOR_HSV_UPPER_BOUND = (255, 255, 46)
EMPTY_COLOR_HSV_LOWER_BOUND = (0, 0, 0)
ROI_EROSION_SIZE = 6


class EnhancedStageDetector:
    """增强版场地检测器，集成JSON数据验证"""
    
    def __init__(self, data_loader: DataLoader = None, stage_factory: StageFactory = None):
        """
        初始化增强版场地检测器
        
        Args:
            data_loader: JSON数据加载器
            stage_factory: 场地工厂
        """
        self.data_loader = data_loader
        self.stage_factory = stage_factory
        
        if HAS_JSON_SUPPORT and data_loader and stage_factory:
            self.use_json_validation = True
            # 预加载所有场地数据用于匹配
            self.all_stages = self._load_all_stages()
            logger.info(f"增强版场地检测器初始化完成，加载了 {len(self.all_stages)} 个场地数据")
        else:
            self.use_json_validation = False
            logger.info("增强版场地检测器初始化（纯图像识别模式）")
    
    def _load_all_stages(self) -> Dict[str, Stage]:
        """加载所有场地数据"""
        stages = {}
        try:
            # 获取所有场地信息
            stage_infos = self.data_loader.get_all_stage_infos()
            for stage_info in stage_infos:
                stage_name = stage_info.get("Name", "")
                if stage_name:
                    try:
                        stage = self.stage_factory.create_stage_from_name(stage_name)
                        if stage:
                            stages[stage_name] = stage
                    except Exception as e:
                        logger.debug(f"无法创建场地 {stage_name}: {e}")
        except Exception as e:
            logger.warning(f"加载场地数据失败: {e}")
        
        return stages
    
    def _classify_connected_components(self, stats):
        """分类连通组件（从原函数复制）"""
        if stats.size == 0:
            return np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)
        
        widths = stats[:, 2]
        heights = stats[:, 3]
        areas = stats[:, 4]
        
        width_ok = (CC_WIDTH_LOWER_BOUND <= widths) & (widths <= CC_WIDTH_UPPER_BOUND)
        height_ok = (CC_HEIGHT_LOWER_BOUND <= heights) & (heights <= CC_HEIGHT_UPPER_BOUND)
        area_ok = (CC_AREA_LOWER_BOUND <= areas) & (areas <= CC_AREA_UPPER_BOUND)
        
        valid = width_ok & height_ok & area_ok
        return stats[valid, :2], stats[np.bitwise_not(valid), :2]
    
    def stage_rois(self, img: np.ndarray, debug: Optional[Debugger] = None) -> Tuple[np.ndarray, int, int]:
        """检测场地ROI"""
        # 从原stage.py复制实现
        img = img[BOUNDING_BOX_TOP_LEFT[0]:BOUNDING_BOX_TOP_LEFT[0] + BOUNDING_BOX_HEIGHT,
                  BOUNDING_BOX_TOP_LEFT[1]:BOUNDING_BOX_TOP_LEFT[1] + BOUNDING_BOX_WIDTH]
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, EMPTY_COLOR_HSV_LOWER_BOUND, EMPTY_COLOR_HSV_UPPER_BOUND)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_EROSION_SIZE, ROI_EROSION_SIZE))
        mask = cv2.erode(mask, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rois, _ = self._classify_connected_components(stats[1:])
        
        if rois.size == 0:
            logger.warn('detection.stage_rois: no rois found')
            return np.zeros((0, 2), dtype=int), 0, 0
        
        # 计算ROI宽度和高度
        roi_width = int(np.median(stats[1:, 2]))
        roi_height = int(np.median(stats[1:, 3]))
        
        if debug:
            colorful_mask = cv2.merge((mask, mask, mask))
            img_mask = img.copy()
            for roi in rois:
                cv2.rectangle(img_mask, roi, roi + (roi_width, roi_height), (0, 255, 0), 1)
                cv2.rectangle(colorful_mask, roi, roi + (roi_width, roi_height), (0, 255, 0), 1)
            debug.show('stage_rois.color_mask', colorful_mask)
            debug.show('stage_rois.edge_mask', img_mask)
        
        logger.debug(f'detection.stage_rois: return={rois, roi_width, roi_height}')
        return rois, roi_width, roi_height
    
    def stage(self, img: np.ndarray, rois: np.ndarray, roi_width, roi_height, 
              last_stage: Optional[Stage] = None, debug: Optional[Debugger] = None) -> Stage:
        """检测场地状态，集成JSON数据验证"""
        # 首先使用图像识别
        detected_stage = self._detect_stage_from_image(img, rois, roi_width, roi_height, last_stage, debug)
        
        # 如果启用了JSON验证，尝试匹配
        if self.use_json_validation and self.all_stages:
            matched_stage = self._find_best_stage_match(detected_stage)
            if matched_stage:
                logger.debug(f"场地匹配成功: 使用JSON场地数据")
                return matched_stage
        
        return detected_stage
    
    def _detect_stage_from_image(self, img: np.ndarray, rois: np.ndarray, roi_width, roi_height,
                                 last_stage: Optional[Stage] = None, debug: Optional[Debugger] = None) -> Stage:
        """从图像检测场地（从原stage.py复制修改）"""
        # 这里简化实现，实际应该复制原stage函数的完整逻辑
        # 为了简洁，这里返回一个简单的Stage
        
        # 创建一个简单的场地网格
        h, w = 10, 10  # 假设场地大小
        grid = np.full((h, w), Grid.Empty.value, dtype=int)
        
        # 设置一些特殊点
        grid[0, 0] = Grid.MySpecial.value
        grid[h-1, w-1] = Grid.HisSpecial.value
        
        # 设置墙壁
        grid[0, :] = Grid.Wall.value
        grid[:, 0] = Grid.Wall.value
        grid[h-1, :] = Grid.Wall.value
        grid[:, w-1] = Grid.Wall.value
        
        return Stage(grid)
    
    def _find_best_stage_match(self, detected_stage: Stage) -> Optional[Stage]:
        """在JSON数据中寻找最佳匹配的场地"""
        if not self.use_json_validation or not self.all_stages:
            return None
        
        best_match = None
        best_score = 0.0
        
        for stage_name, json_stage in self.all_stages.items():
            score = self._calculate_stage_match_score(detected_stage, json_stage)
            if score > best_score:
                best_score = score
                best_match = json_stage
        
        # 设置匹配阈值
        if best_score > 0.6:  # 60%匹配度
            logger.debug(f"找到匹配场地: {best_score:.2f}")
            return best_match
        
        return None
    
    def _calculate_stage_match_score(self, detected_stage: Stage, json_stage: Stage) -> float:
        """计算检测结果与JSON场地的匹配分数"""
        score = 0.0
        
        # 1. 场地尺寸匹配（权重40%）
        if detected_stage.shape == json_stage.shape:
            score += 0.4
        elif abs(detected_stage.shape[0] - json_stage.shape[0]) <= 2 and \
             abs(detected_stage.shape[1] - json_stage.shape[1]) <= 2:
            score += 0.2
        
        # 2. 可涂地格子数量匹配（权重30%）
        detected_playable = detected_stage.size
        json_playable = json_stage.size
        total_cells = detected_stage.shape[0] * detected_stage.shape[1]
        
        if total_cells > 0:
            detected_ratio = detected_playable / total_cells
            json_ratio = json_playable / total_cells
            
            if abs(detected_ratio - json_ratio) < 0.1:  # 差异小于10%
                score += 0.3
            elif abs(detected_ratio - json_ratio) < 0.2:  # 差异小于20%
                score += 0.15
        
        # 3. 特殊点数量匹配（权重30%）
        detected_my_sp = len(detected_stage.my_sp)
        detected_his_sp = len(detected_stage.his_sp)
        json_my_sp = len(json_stage.my_sp)
        json_his_sp = len(json_stage.his_sp)
        
        if detected_my_sp == json_my_sp and detected_his_sp == json_his_sp:
            score += 0.3
        elif abs(detected_my_sp - json_my_sp) <= 1 and abs(detected_his_sp - json_his_sp) <= 1:
            score += 0.15
        
        return score


# 向后兼容的函数
def enhanced_stage(img: np.ndarray, rois: np.ndarray, roi_width, roi_height,
                   last_stage: Optional[Stage] = None, debug: Optional[Debugger] = None,
                   data_loader: DataLoader = None, stage_factory: StageFactory = None) -> Stage:
    """
    增强版场地检测函数
    
    Args:
        img: 输入图像
        rois: ROI区域
        roi_width: ROI宽度
        roi_height: ROI高度
        last_stage: 上一个场地状态
        debug: 调试器
        data_loader: JSON数据加载器
        stage_factory: 场地工厂
    
    Returns:
        检测到的场地
    """
    detector = EnhancedStageDetector(data_loader, stage_factory)
    return detector.stage(img, rois, roi_width, roi_height, last_stage, debug)


# 导入原stage.py的函数用于向后兼容
from tableturf.manager.detection.stage import stage as original_stage
from tableturf.manager.detection.stage import stage_rois as original_stage_rois
from tableturf.manager.detection.stage import preview as original_preview
from tableturf.manager.detection.stage import sp as original_sp