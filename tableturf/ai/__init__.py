from tableturf.ai.interface import AI
from tableturf.ai.simple import SimpleAI

__all__ = ['AI', 'SimpleAI']

# 导入Alpha AI
try:
    from tableturf.ai.alpha.alpha import Alpha
    from tableturf.ai.alpha.enhanced_alpha import EnhancedAlpha
    __all__.extend(['Alpha', 'EnhancedAlpha'])
except ImportError:
    pass

# 导入MCTS AI
try:
    from tableturf.ai.mcts import MCTSAI
    __all__.append('MCTSAI')
except ImportError:
    pass
