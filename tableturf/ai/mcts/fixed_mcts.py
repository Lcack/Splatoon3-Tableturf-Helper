import math
import random
import copy
import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime

from tableturf.model import Status, Step, Grid
from tableturf.ai import AI
from tableturf.ai.alpha import util
from logger import logger

class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, state: Status, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        # 获取所有合法动作
        self.untried_actions = self.state.get_possible_steps()
    
    def is_fully_expanded(self) -> bool:
        """检查是否完全扩展"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """检查是否为终止节点（游戏结束）"""
        return self.state.round <= 0
    
    def uct_select_child(self, exploration_weight=1.41):
        """使用 UCB1 算法选择子节点"""
        if not self.children:
            return None
        
        # 只考虑有访问次数的子节点
        valid_children = [c for c in self.children if c.visits > 0]
        if not valid_children:
            return random.choice(self.children)
        
        return max(valid_children, key=lambda c: 
                   (c.value / c.visits) + 
                   exploration_weight * math.sqrt(math.log(self.visits) / c.visits))

class MCTSAI(AI):
    """MCTS AI实现"""
    
    def __init__(self, simulation_limit=200, time_limit_ms=1000):
        """
        MCTS AI实现
        
        Args:
            simulation_limit: 最大模拟次数
            time_limit_ms: 最大思考时间（毫秒）
        """
        self.simulation_limit = simulation_limit
        self.time_limit_ms = time_limit_ms
        self.exploration_weight = 1.41
        
        # 学习权重（可选）
        self.weights = {
            "my_ink": 1.0,
            "his_ink": -0.8,
            "my_sp": 5.0,
            "his_sp": -4.0,
            "connectivity": 0.3,
            "special_points": 2.0,
            "round_bonus": 0.1
        }
        
        logger.info(f"MCTSAI初始化: 模拟次数={simulation_limit}, 时间限制={time_limit_ms}ms")
    
    def redraw(self, hands, stage=None, my_deck=None, his_deck=None):
        """重抽决策 - 使用简单启发式"""
        # 如果手牌平均面积小于7，重抽
        avg_size = sum(card.size for card in hands) / len(hands)
        result = avg_size < 7
        logger.debug(f"MCTS重抽决策: 平均面积={avg_size:.1f}, 重抽={result}")
        return result
    
    def next_step(self, status: Status) -> Step:
        """使用MCTS选择最佳行动"""
        start_time = datetime.now()
        
        # 创建根节点
        root = MCTSNode(copy.deepcopy(status))
        
        # MCTS搜索
        simulations = 0
        for i in range(self.simulation_limit):
            simulations = i + 1
            
            # 检查时间限制
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms > self.time_limit_ms:
                logger.debug(f"MCTS时间限制到达: {elapsed_ms:.0f}ms, 模拟次数: {simulations}")
                break
            
            # 选择 -> 扩展 -> 模拟 -> 回溯
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
        
        # 选择最佳行动（访问次数最多）
        if not root.children:
            # 如果没有子节点，返回随机合法行动
            steps = status.get_possible_steps()
            if not steps:
                logger.warning("MCTS: 没有合法行动")
                # 创建跳过行动
                if status.hands:
                    return Step(Step.Action.Skip, status.hands[0], None, None)
                else:
                    # 紧急情况：返回任意行动
                    return Step(Step.Action.Skip, None, None, None)
            
            step = random.choice(steps)
            logger.debug(f"MCTS随机决策: {step}")
            return step
        
        best_child = max(root.children, key=lambda c: c.visits)
        logger.debug(f"MCTS决策: 模拟次数={simulations}, 访问次数={best_child.visits}, 价值={best_child.value:.2f}")
        return best_child.action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段：使用UCT选择节点"""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                child = node.uct_select_child(self.exploration_weight)
                if child is None:
                    # 如果没有有效的子节点，返回当前节点
                    return node
                node = child
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展阶段：扩展一个未尝试的行动"""
        if node.is_fully_expanded():
            return node
        
        # 随机选择一个未尝试的行动
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # 模拟执行行动
        try:
            next_states = util.estimate_status(node.state, action, expand=True)
            if next_states:
                next_state = random.choice(next_states)
                child_node = MCTSNode(next_state, parent=node, action=action)
                node.children.append(child_node)
                return child_node
            else:
                # 如果没有下一个状态，返回当前节点
                return node
        except Exception as e:
            logger.warning(f"MCTS扩展失败: {e}")
            return node
    
    def _simulate(self, state: Status) -> float:
        """模拟阶段：快速走子直到游戏结束"""
        current_state = copy.deepcopy(state)
        
        # 快速走子策略
        while current_state.round > 0:
            # 获取所有合法行动
            possible_steps = current_state.get_possible_steps()
            if not possible_steps:
                break
            
            # 使用启发式选择行动
            step = self._heuristic_select(current_state, possible_steps)
            if step is None:
                break
            
            # 应用行动
            try:
                next_states = util.estimate_status(current_state, step, expand=True)
                if next_states:
                    current_state = random.choice(next_states)
                else:
                    break
            except Exception as e:
                logger.debug(f"模拟阶段错误: {e}")
                break
        
        # 计算奖励
        return self._calculate_reward(current_state)
    
    def _heuristic_select(self, state: Status, possible_steps: List[Step]) -> Step:
        """启发式行动选择"""
        if not possible_steps:
            return None
        
        # 80%概率使用启发式，20%概率随机
        if random.random() < 0.8:
            # 使用简单评分函数
            scores = []
            for step in possible_steps:
                try:
                    next_states = util.estimate_status(state, step, expand=False)
                    if next_states:
                        next_state = next_states[0]
                        score = self._evaluate_state(next_state)
                        scores.append(score)
                    else:
                        scores.append(-1000)
                except:
                    scores.append(-1000)
            
            if scores and max(scores) > -1000:
                return possible_steps[np.argmax(scores)]
        
        # 随机选择
        return random.choice(possible_steps)
    
    def _evaluate_state(self, state: Status) -> float:
        """状态评估函数"""
        # 基础评分
        score = 0
        
        # 墨迹数量
        my_ink = len(state.stage.my_ink)
        his_ink = len(state.stage.his_ink)
        score += my_ink * self.weights["my_ink"]
        score += his_ink * self.weights["his_ink"]
        
        # SP点数
        score += state.my_sp * self.weights["my_sp"]
        score += state.his_sp * self.weights["his_sp"]
        
        # 特殊点数量
        my_special = len(state.stage.my_sp)
        his_special = len(state.stage.his_sp)
        score += (my_special - his_special) * self.weights["special_points"]
        
        # 回合数奖励（早期更注重发展）
        round_bonus = state.round * self.weights["round_bonus"]
        score += round_bonus
        
        # 连通性（简单估计）
        connectivity = len(state.get_possible_steps()) / 100.0
        score += connectivity * self.weights["connectivity"]
        
        return score
    
    def _calculate_reward(self, state: Status) -> float:
        """计算最终奖励"""
        my_ink = len(state.stage.my_ink)
        his_ink = len(state.stage.his_ink)
        
        if my_ink > his_ink:
            return 1.0  # 胜利
        elif my_ink < his_ink:
            return 0.0  # 失败
        else:
            return 0.5  # 平局
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """回溯阶段：更新节点统计"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def reset(self):
        """重置AI状态"""
        pass

class EnhancedMCTSAI(MCTSAI):
    """增强版MCTS AI，集成JSON数据"""
    
    def __init__(self, npc_id: Optional[str] = None, level: int = 0, **kwargs):
        """
        增强版MCTS AI
        
        Args:
            npc_id: 对战的NPC ID
            level: NPC等级
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        
        # 导入数据模块
        from tableturf.model import DataLoader, CardFactory, StageFactory
        
        self.data_loader = DataLoader()
        self.card_factory = CardFactory(self.data_loader)
        self.stage_factory = StageFactory(self.data_loader)
        
        # NPC数据
        self.npc_id = npc_id
        self.npc_level = level
        self.npc_info = None
        self.npc_deck = None
        self.npc_stage = None
        
        # 加载NPC数据
        if npc_id:
            self._load_npc_data(npc_id, level)
    
    def _load_npc_data(self, npc_id: str, level: int = 0):
        """加载NPC数据"""
        self.npc_info = self.data_loader.get_npc_by_id(npc_id)
        if not self.npc_info:
            logger.warning(f"未找到NPC: {npc_id}")
            return
        
        logger.info(f"加载NPC: {self.npc_info.get('Name', 'Unknown')} "
                   f"({self.npc_info.get('name_zh', 'Unknown')})")
        
        # 加载NPC卡组
        self.npc_deck = self.card_factory.create_deck_from_npc_id(npc_id)
        logger.info(f"NPC卡组: {len(self.npc_deck)} 张卡牌")
        
        # 加载NPC场地
        map_list = self.npc_info.get("Map", [])
        if map_list and level < len(map_list):
            stage_name = map_list[level]
            self.npc_stage = self.stage_factory.create_stage_from_name(stage_name)
            if self.npc_stage:
                logger.info(f"NPC场地: {stage_name} ({self.npc_stage.shape[0]}x{self.npc_stage.shape[1]})")
    
    def analyze_npc_strategy(self):
        """分析NPC策略特点"""
        if not self.npc_info:
            return None
        
        analysis = {
            "npc_id": self.npc_id,
            "name": self.npc_info.get("Name", ""),
            "name_zh": self.npc_info.get("name_zh", ""),
            "ai_levels": self.npc_info.get("AILevel", []),
            "ai_types": self.npc_info.get("AIType", []),
            "maps": self.npc_info.get("Map", []),
            "deck_size": len(self.npc_deck) if self.npc_deck else 0,
        }
        
        # 卡组分析
        if self.npc_deck:
            analysis["deck_analysis"] = self._analyze_deck(self.npc_deck)
        
        # 场地分析
        if self.npc_stage:
            analysis["stage_analysis"] = self._analyze_stage(self.npc_stage)
        
        return analysis
    
    def _analyze_deck(self, deck):
        """分析卡组特点"""
        if not deck:
            return {}
        
        sizes = [card.size for card in deck]
        sp_costs = [card.sp_cost for card in deck]
        special_cards = [card for card in deck if card.sp_cost > 0]
        
        return {
            "total_cards": len(deck),
            "avg_size": np.mean(sizes),
            "min_size": np.min(sizes),
            "max_size": np.max(sizes),
            "avg_sp_cost": np.mean(sp_costs),
            "special_cards_count": len(special_cards),
            "special_cards_ratio": len(special_cards) / len(deck),
            "size_std": np.std(sizes)
        }
    
    def _analyze_stage(self, stage):
        """分析场地特点"""
        return {
            "dimensions": stage.shape,
            "playable_cells": stage.size,
            "my_special_points": len(stage.my_sp),
            "his_special_points": len(stage.his_sp),
            "playable_percentage": (stage.size / (stage.shape[0] * stage.shape[1])) * 100
        }
    
    def next_step(self, status: Status) -> Step:
        """增强版决策，考虑NPC数据"""
        # 如果有NPC数据，调整MCTS参数
        if self.npc_info and self.npc_level < len(self.npc_info.get("AIType", [])):
            ai_type = self.npc_info["AIType"][self.npc_level]
            
            # 根据AI类型调整探索权重
            if ai_type == "Aggressive":
                # 激进型：减少探索，增加利用
                self.exploration_weight = 1.0
                logger.debug(f"MCTS调整: 激进型AI, 探索权重={self.exploration_weight}")
            elif ai_type == "Defensive":
                # 防御型：增加探索
                self.exploration_weight = 2.0
                logger.debug(f"MCTS调整: 防御型AI, 探索权重={self.exploration_weight}")
        
        # 调用父类MCTS决策
        return super().next_step(status)
    
    def get_recommended_counter_npc(self):
        """获取推荐克制的NPC"""
        if not self.npc_info or not self.npc_deck:
            return []
        
        # 分析当前NPC卡组
        deck_analysis = self._analyze_deck(self.npc_deck)
        target_avg_size = deck_analysis.get("avg_size", 0)
        target_special_ratio = deck_analysis.get("special_cards_ratio", 0)
        
        # 寻找所有NPC
        all_npc_ids = self.data_loader.get_all_npc_ids()
        counter_candidates = []
        
        for npc_id in all_npc_ids:
            if npc_id == self.npc_id:
                continue
            
            # 加载NPC数据
            npc_info = self.data_loader.get_npc_by_id(npc_id)
            if not npc_info:
                continue
            
            deck = self.card_factory.create_deck_from_npc_id(npc_id)
            if not deck:
                continue
            
            # 分析卡组
            analysis = self._analyze_deck(deck)
            avg_size = analysis.get("avg_size", 0)
            special_ratio = analysis.get("special_cards_ratio", 0)
            
            # 克制评分
            size_advantage = avg_size - target_avg_size
            special_advantage = special_ratio - target_special_ratio
            
            # 综合评分
            score = size_advantage * 0.7 + special_advantage * 0.3
            
            if score > 0.5:  # 有明显优势
                counter_candidates.append((
                    npc_id,
                    score,
                    npc_info.get("Name", "Unknown"),
                    npc_info.get("name_zh", ""),
                    avg_size,
                    special_ratio
                ))
        
        # 按评分排序
        counter_candidates.sort(key=lambda x: x[1], reverse=True)
        return counter_candidates
    
    def reset(self):
        """重置AI状态"""
        super().reset()
        # 可以在这里重置增强版特有的状态