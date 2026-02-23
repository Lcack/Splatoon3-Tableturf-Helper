"""
增强版卡牌检测模块
集成JSON数据验证，提高识别准确性
"""

from typing import List, Optional, Dict, Tuple
import cv2
import numpy as np
from logger import logger
from tableturf.manager.detection.debugger import Debugger
from tableturf.manager.detection import util
from tableturf.manager.detection.ui import hands_cursor
from tableturf.model import Card, Grid, Pattern

# 导入JSON数据模块
try:
    from tableturf.model import DataLoader, CardFactory
    HAS_JSON_SUPPORT = True
except ImportError:
    HAS_JSON_SUPPORT = False
    logger.warning("JSON数据模块不可用，将使用纯图像识别")

# 从原card.py复制常量
HANDS_GRID_TOP_LEFTS = np.array([[192, 51], [203, 305], [516, 51], [516, 305]])
HANDS_GRID_ROI_WIDTH = 14
HANDS_GRID_ROI_HEIGHT = 14
HANDS_GRID_ROI_WIDTH_STEPS = [25.9, 24.7, 25.9, 24.7]
HANDS_GRID_ROI_HEIGHT_STEPS = [23.9, 23, 23.9, 23]
HANDS_GRID_ROI_HEIGHT_OFFSETS = [1, 0.8, 0, 0]
HANDS_GRID_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 8, 8, HANDS_GRID_ROI_WIDTH_STEPS[i], HANDS_GRID_ROI_HEIGHT_STEPS[i], 0, HANDS_GRID_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(HANDS_GRID_TOP_LEFTS)]).reshape((4, 64, 2))

HANDS_COST_TOP_LEFTS = np.array([[410, 123], [414, 373], [731, 123], [725, 373]])
HANDS_COST_ROI_WIDTH = 12
HANDS_COST_ROI_HEIGHT = 12
HANDS_COST_ROI_WIDTH_STEPS = [22.9, 21.5, 22.9, 21.5]
HANDS_COST_ROI_HEIGHT_OFFSETS = [0, 0, -0.6, -0.6]
HANDS_COST_OPENCV_ROI_LEFT_TOP = np.array([util.numpy_to_opencv(idx) for idx in HANDS_COST_TOP_LEFTS])  # shape: (4, 2)
HANDS_COST_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 6, 1, HANDS_COST_ROI_WIDTH_STEPS[i], 0, 0, HANDS_COST_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(HANDS_COST_TOP_LEFTS)]).reshape((4, 6, 2))

FOCUS_GRID_TOP_LEFTS = np.array([[192, 42], [200, 295], [514, 43], [515, 295]])
FOCUS_GRID_ROI_WIDTH = 14
FOCUS_GRID_ROI_HEIGHT = 14
FOCUS_GRID_ROI_WIDTH_STEPS = [26.9, 25.7, 26.9, 25.7]
FOCUS_GRID_ROI_HEIGHT_STEPS = [24.9, 24, 24.9, 24]
FOCUS_GRID_ROI_WIDTH_OFFSETS = [1, 1, 1, 1]
FOCUS_GRID_ROI_HEIGHT_OFFSETS = [-0.6, -0.4, -1, -1]
FOCUS_GRID_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 8, 8, FOCUS_GRID_ROI_WIDTH_STEPS[i], FOCUS_GRID_ROI_HEIGHT_STEPS[i], FOCUS_GRID_ROI_WIDTH_OFFSETS[i], FOCUS_GRID_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(FOCUS_GRID_TOP_LEFTS)]).reshape((4, 64, 2))

FOCUS_COST_TOP_LEFTS = np.array([[415, 128], [417, 375], [735, 127], [730, 376]])
FOCUS_COST_ROI_WIDTH = 12
FOCUS_COST_ROI_HEIGHT = 12
FOCUS_COST_ROI_WIDTH_STEPS = [22.9, 22.5, 23.9, 22.5]
FOCUS_COST_ROI_HEIGHT_OFFSETS = [-0.5, 0, -1, -1]
FOCUS_COST_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 6, 1, FOCUS_COST_ROI_WIDTH_STEPS[i], 0, 0, FOCUS_COST_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(FOCUS_COST_TOP_LEFTS)]).reshape((4, 6, 2))

MY_INK_LIGHTER_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
MY_INK_LIGHTER_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
MY_SPECIAL_LIGHTER_COLOR_HSV_UPPER_BOUND = (25, 255, 255)
MY_SPECIAL_LIGHTER_COLOR_HSV_LOWER_BOUND = (20, 100, 150)
MY_INK_DARKER_COLOR_HSV_UPPER_BOUND = (35, 255, 140)
MY_INK_DARKER_COLOR_HSV_LOWER_BOUND = (27, 100, 100)
MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND = (27, 255, 140)
MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND = (15, 100, 100)
GRID_PIXEL_RATIO = 0.3


class EnhancedCardDetector:
    """增强版卡牌检测器，集成JSON数据验证"""
    
    def __init__(self, data_loader: DataLoader = None, card_factory: CardFactory = None):
        """
        初始化增强版检测器
        
        Args:
            data_loader: JSON数据加载器
            card_factory: 卡牌工厂
        """
        self.data_loader = data_loader
        self.card_factory = card_factory
        
        if HAS_JSON_SUPPORT and data_loader and card_factory:
            self.use_json_validation = True
            # 预加载所有卡牌数据用于匹配
            self.all_cards = self._load_all_cards()
            logger.info(f"增强版卡牌检测器初始化完成，加载了 {len(self.all_cards)} 张卡牌数据")
        else:
            self.use_json_validation = False
            logger.info("增强版卡牌检测器初始化（纯图像识别模式）")
    
    def _load_all_cards(self) -> Dict[str, Card]:
        """加载所有卡牌数据"""
        cards = {}
        try:
            # 获取所有卡牌ID
            card_infos = self.data_loader.get_all_card_infos()
            for card_id, card_info in card_infos.items():
                try:
                    card = self.card_factory.create_card_from_id(card_id)
                    if card:
                        cards[card_id] = card
                except Exception as e:
                    logger.debug(f"无法创建卡牌 {card_id}: {e}")
        except Exception as e:
            logger.warning(f"加载卡牌数据失败: {e}")
        
        return cards
    
    def _grid_ratios(self, img: np.ndarray, top_left: np.ndarray, lower_bound, upper_bound) -> float:
        """计算格子颜色比例（从原函数复制）"""
        roi = img[top_left[0]:top_left[0] + HANDS_GRID_ROI_HEIGHT, 
                  top_left[1]:top_left[1] + HANDS_GRID_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return np.sum(mask == 255) / (HANDS_GRID_ROI_WIDTH * HANDS_GRID_ROI_HEIGHT)
    
    def _find_best_match(self, detected_grid: np.ndarray, detected_cost: int) -> Tuple[Optional[Card], float]:
        """在JSON数据中寻找最佳匹配的卡牌"""
        if not self.use_json_validation or not self.all_cards:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for card_id, json_card in self.all_cards.items():
            # 计算匹配分数
            score = self._calculate_match_score(detected_grid, detected_cost, json_card)
            
            if score > best_score:
                best_score = score
                best_match = json_card
        
        # 设置匹配阈值
        if best_score > 0.7:  # 70%匹配度
            logger.debug(f"找到匹配卡牌: 分数={best_score:.2f}")
            return best_match, best_score
        
        return None, best_score
    
    def _calculate_match_score(self, detected_grid: np.ndarray, detected_cost: int, json_card: Card) -> float:
        """计算检测结果与JSON卡牌的匹配分数"""
        score = 0.0
        
        # 1. SP消耗匹配（权重40%）
        if json_card.sp_cost == detected_cost:
            score += 0.4
        elif abs(json_card.sp_cost - detected_cost) <= 1:
            score += 0.2
        
        # 2. 卡牌大小匹配（权重30%）
        detected_size = np.sum(detected_grid != Grid.Empty.value)
        json_size = json_card.size
        if json_size == detected_size:
            score += 0.3
        elif abs(json_size - detected_size) <= 2:
            score += 0.15
        
        # 3. 图案形状匹配（权重30%）
        # 将检测到的网格转换为Pattern进行比较
        try:
            detected_pattern = Pattern(detected_grid)
            json_pattern = json_card.pattern
            
            # 比较基本属性
            if detected_pattern.width == json_pattern.width and detected_pattern.height == json_pattern.height:
                score += 0.15
            
            # 简单形状匹配（可以扩展为更复杂的形状匹配）
            if detected_pattern.size == json_pattern.size:
                score += 0.15
        
        except Exception as e:
            logger.debug(f"图案匹配失败: {e}")
        
        return score
    
    def hands(self, img: np.ndarray, cursor=None, debug: Optional[Debugger] = None) -> List[Card]:
        """
        增强版手牌检测
        集成JSON数据验证，提高准确性
        """
        # 第一步：图像识别（使用原有算法）
        if cursor is None:
            cursor = hands_cursor(img, debug)
        
        grid_rois = HANDS_GRID_NUMPY_ROI_TOP_LEFTS.copy()
        cost_rois = HANDS_COST_NUMPY_ROI_TOP_LEFTS.copy()
        if 0 <= cursor < 4:
            grid_rois[cursor] = FOCUS_GRID_NUMPY_ROI_TOP_LEFTS[cursor]
            cost_rois[cursor] = FOCUS_COST_NUMPY_ROI_TOP_LEFTS[cursor]
        
        ink_lower_bound, ink_upper_bound = MY_INK_LIGHTER_COLOR_HSV_LOWER_BOUND, MY_INK_LIGHTER_COLOR_HSV_UPPER_BOUND
        special_lower_bound, special_upper_bound = MY_SPECIAL_LIGHTER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_LIGHTER_COLOR_HSV_UPPER_BOUND
        
        # 计算颜色比例
        grid_ink_ratios = np.array([self._grid_ratios(img, idx, ink_lower_bound, ink_upper_bound) 
                                   for grid in grid_rois for idx in grid]).reshape(4, 64)
        grid_special_ratios = np.array([self._grid_ratios(img, idx, special_lower_bound, special_upper_bound) 
                                       for grid in grid_rois for idx in grid]).reshape(4, 64)
        dark_grid_ink_ratios = np.array([self._grid_ratios(img, idx, MY_INK_DARKER_COLOR_HSV_LOWER_BOUND, MY_INK_DARKER_COLOR_HSV_UPPER_BOUND) 
                                        for grid in grid_rois for idx in grid]).reshape(4, 64)
        dark_grid_special_ratios = np.array([self._grid_ratios(img, idx, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND) 
                                           for grid in grid_rois for idx in grid]).reshape(4, 64)
        
        cost_ratios = np.array([self._grid_ratios(img, idx, special_lower_bound, special_upper_bound) 
                               for grid in cost_rois for idx in grid]).reshape(4, 6)
        dark_cost_ratios = np.array([self._grid_ratios(img, idx, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND) 
                                    for grid in cost_rois for idx in grid]).reshape(4, 6)
        cost_ratios = np.maximum(cost_ratios, dark_cost_ratios)
        
        # 生成网格
        grids = np.full((4, 64), Grid.Empty.value, dtype=int)
        grids[grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
        grids[grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
        grids[dark_grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
        grids[dark_grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
        costs = np.sum(cost_ratios > GRID_PIXEL_RATIO, axis=1)
        
        grids = grids.reshape((4, 8, 8))
        
        # 第二步：JSON数据验证（如果可用）
        cards = []
        validation_stats = {"matched": 0, "total": 4}
        
        for i in range(4):
            detected_grid = grids[i]
            detected_cost = costs[i]
            
            if self.use_json_validation:
                # 尝试在JSON数据中寻找匹配
                matched_card, match_score = self._find_best_match(detected_grid, detected_cost)
                
                if matched_card and match_score > 0.7:
                    # 使用JSON中的卡牌
                    cards.append(matched_card)
                    validation_stats["matched"] += 1
                    logger.debug(f"卡牌{i}: 使用JSON匹配 (分数={match_score:.2f})")
                else:
                    # 使用图像识别结果
                    card = Card(detected_grid, detected_cost)
                    cards.append(card)
                    if match_score > 0:
                        logger.debug(f"卡牌{i}: 图像识别 (JSON匹配分数={match_score:.2f})")
                    else:
                        logger.debug(f"卡牌{i}: 图像识别 (无JSON匹配)")
            else:
                # 纯图像识别
                card = Card(detected_grid, detected_cost)
                cards.append(card)
        
        # 记录验证统计
        if self.use_json_validation:
            match_rate = validation_stats["matched"] / validation_stats["total"] * 100
            logger.info(f"卡牌验证完成: {validation_stats['matched']}/{validation_stats['total']} 匹配 ({match_rate:.1f}%)")
        
        # 调试输出（从原函数复制）
        if debug:
            img2 = img.copy()
            grid_rois_opencv = np.array([util.numpy_to_opencv(idx) for grid in grid_rois for idx in grid]).reshape((4, 64, 2))
            cost_rois_opencv = np.array([util.numpy_to_opencv(idx) for grid in cost_rois for idx in grid]).reshape((4, 6, 2))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ink_mask = cv2.inRange(hsv, ink_lower_bound, ink_upper_bound)
            special_mask = cv2.inRange(hsv, special_lower_bound, special_upper_bound)
            dark_ink_mask = cv2.inRange(hsv, MY_INK_DARKER_COLOR_HSV_LOWER_BOUND, MY_INK_DARKER_COLOR_HSV_UPPER_BOUND)
            dark_special_mask = cv2.inRange(hsv, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND)
            mask = np.max([ink_mask, special_mask, dark_ink_mask, dark_special_mask], axis=0)
            mask = cv2.merge((mask, mask, mask))
            
            for i, grid in enumerate(grid_rois_opencv):
                for k, roi in enumerate(grid):
                    cv2.rectangle(img2, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                    cv2.putText(img2, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                    cv2.putText(mask, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    
                    grid_value = grids[i][k // 8][k % 8]
                    if grid_value == Grid.MyInk.value:
                        cv2.rectangle(img2, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 0, 255), 1)
                        cv2.putText(img2, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                        cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 0, 255), 1)
                        cv2.putText(mask, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    elif grid_value == Grid.MySpecial.value:
                        cv2.rectangle(img2, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (255, 0, 0), 1)
                        cv2.putText(img2, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                        cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (255, 0, 0), 1)
                        cv2.putText(mask, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            for i, cost in enumerate(cost_rois_opencv):
                for k, roi in enumerate(cost):
                    cv2.rectangle(img2, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (0, 255, 0), 1)
                    cv2.rectangle(mask, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (0, 255, 0), 1)
                    if cost_ratios[i][k] > GRID_PIXEL_RATIO:
                        cv2.rectangle(img2, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (255, 0, 0), 1)
                        cv2.rectangle(mask, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (255, 0, 0), 1)
                cv2.putText(mask, f'{costs[i]}', HANDS_COST_OPENCV_ROI_LEFT_TOP[i] + np.rint([-HANDS_GRID_ROI_WIDTH * 1.5, HANDS_COST_ROI_HEIGHT]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            debug.show('enhanced_hands.image', img2)
            debug.show('enhanced_hands.color_mask', mask)
        
        logger.debug(f'enhanced_detection.hands: return={cards}')
        return cards


# 向后兼容的函数
def enhanced_hands(img: np.ndarray, cursor=None, debug: Optional[Debugger] = None, 
                   data_loader: DataLoader = None, card_factory: CardFactory = None) -> List[Card]:
    """
    增强版手牌检测函数（向后兼容）
    
    Args:
        img: 输入图像
        cursor: 光标位置
        debug: 调试器
        data_loader: JSON数据加载器
        card_factory: 卡牌工厂
    
    Returns:
        检测到的卡牌列表
    """
    detector = EnhancedCardDetector(data_loader, card_factory)
    return detector.hands(img, cursor, debug)


# 完全替代原card.py的hands函数
def hands(img: np.ndarray, cursor=None, debug: Optional[Debugger] = None) -> List[Card]:
    """
    完全替代原card.py的hands函数
    如果JSON数据可用，使用增强版检测；否则使用基础检测
    
    Args:
        img: 输入图像
        cursor: 光标位置
        debug: 调试器
    
    Returns:
        检测到的卡牌列表
    """
    try:
        # 尝试使用增强版检测
        from tableturf.model import DataLoader, CardFactory
        loader = DataLoader()
        factory = CardFactory(loader)
        return enhanced_hands(img, cursor, debug, loader, factory)
    except Exception as e:
        # 如果增强版失败，使用基础检测逻辑
        logger.warning(f"增强版检测失败，使用基础检测: {e}")
        return _fallback_hands(img, cursor, debug)


def _fallback_hands(img: np.ndarray, cursor=None, debug: Optional[Debugger] = None) -> List[Card]:
    """
    基础检测逻辑（从原card.py复制修改）
    """
    def __grid_ratios(top_left: np.ndarray, lower_bound, upper_bound) -> float:
        roi = img[top_left[0]:top_left[0] + HANDS_GRID_ROI_HEIGHT, 
                  top_left[1]:top_left[1] + HANDS_GRID_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return np.sum(mask == 255) / (HANDS_GRID_ROI_WIDTH * HANDS_GRID_ROI_HEIGHT)
    
    if cursor is None:
        cursor = hands_cursor(img, debug)
    
    grid_rois = HANDS_GRID_NUMPY_ROI_TOP_LEFTS.copy()
    cost_rois = HANDS_COST_NUMPY_ROI_TOP_LEFTS.copy()
    if 0 <= cursor < 4:
        grid_rois[cursor] = FOCUS_GRID_NUMPY_ROI_TOP_LEFTS[cursor]
        cost_rois[cursor] = FOCUS_COST_NUMPY_ROI_TOP_LEFTS[cursor]
    
    ink_lower_bound, ink_upper_bound = MY_INK_LIGHTER_COLOR_HSV_LOWER_BOUND, MY_INK_LIGHTER_COLOR_HSV_UPPER_BOUND
    special_lower_bound, special_upper_bound = MY_SPECIAL_LIGHTER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_LIGHTER_COLOR_HSV_UPPER_BOUND
    
    grid_ink_ratios = np.array([__grid_ratios(idx, ink_lower_bound, ink_upper_bound) 
                               for grid in grid_rois for idx in grid]).reshape(4, 64)
    grid_special_ratios = np.array([__grid_ratios(idx, special_lower_bound, special_upper_bound) 
                                   for grid in grid_rois for idx in grid]).reshape(4, 64)
    dark_grid_ink_ratios = np.array([__grid_ratios(idx, MY_INK_DARKER_COLOR_HSV_LOWER_BOUND, MY_INK_DARKER_COLOR_HSV_UPPER_BOUND) 
                                    for grid in grid_rois for idx in grid]).reshape(4, 64)
    dark_grid_special_ratios = np.array([__grid_ratios(idx, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND) 
                                       for grid in grid_rois for idx in grid]).reshape(4, 64)
    
    cost_ratios = np.array([__grid_ratios(idx, special_lower_bound, special_upper_bound) 
                           for grid in cost_rois for idx in grid]).reshape(4, 6)
    dark_cost_ratios = np.array([__grid_ratios(idx, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND) 
                                for grid in cost_rois for idx in grid]).reshape(4, 6)
    cost_ratios = np.maximum(cost_ratios, dark_cost_ratios)
    
    grids = np.full((4, 64), Grid.Empty.value, dtype=int)
    grids[grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
    grids[grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
    grids[dark_grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
    grids[dark_grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
    costs = np.sum(cost_ratios > GRID_PIXEL_RATIO, axis=1)
    
    grids = grids.reshape((4, 8, 8))
    cards = [Card(grids[i], costs[i]) for i in range(4)]
    
    logger.debug(f'detection.hands: return={cards}')
    return cards