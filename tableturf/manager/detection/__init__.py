# 使用增强版card检测完全替代原card.py
from tableturf.manager.detection.enhanced_card import hands, enhanced_hands, EnhancedCardDetector

from tableturf.manager.detection.deck import deck
from tableturf.manager.detection.stage import stage_rois, stage, preview, sp
from tableturf.manager.detection.ui import deck_cursor, hands_cursor, redraw_cursor, special_on, skip, replay_cursor, giveup_cursor, result, level, start

# 导入增强版stage检测
try:
    from tableturf.manager.detection.enhanced_stage import enhanced_stage, EnhancedStageDetector
except ImportError:
    # 如果增强版不可用，使用原版
    enhanced_stage = stage
    EnhancedStageDetector = None
