from tableturf.ai.interface import AI
from tableturf.ai.simple import SimpleAI

# 导入Alpha AI
try:
    from tableturf.ai.alpha.alpha import Alpha
    from tableturf.ai.alpha.enhanced_alpha import EnhancedAlpha
except ImportError:
    # Alpha模块可能不可用
    pass

# 导入MCTS AI
try:
    from tableturf.ai.mcts import MCTSAI, EnhancedMCTSAI
except ImportError:
    # MCTS模块可能不可用
    pass

__all__ = ['AI', 'SimpleAI', 'Alpha', 'EnhancedAlpha', 'MCTSAI', 'EnhancedMCTSAI']
