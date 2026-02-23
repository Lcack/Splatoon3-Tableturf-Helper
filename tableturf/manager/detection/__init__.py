from tableturf.manager.detection.card import hands
from tableturf.manager.detection.deck import deck
from tableturf.manager.detection.stage import stage_rois, stage, preview, sp
from tableturf.manager.detection.ui import deck_cursor, hands_cursor, redraw_cursor, special_on, skip, replay_cursor, giveup_cursor, result, level, start

# 导入增强版检测模块
try:
    from tableturf.manager.detection.enhanced_card import enhanced_hands, EnhancedCardDetector
except ImportError:
    # 如果增强版不可用，提供向后兼容
    enhanced_hands = hands
    EnhancedCardDetector = None
