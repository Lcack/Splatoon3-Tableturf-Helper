#!/usr/bin/env python3
"""
增强版Alpha AI
集成JSON数据支持的Alpha算法
"""

import logging
from typing import List, Optional

import numpy as np

from logger import logger
from tableturf.ai import AI
from tableturf.ai.alpha import util
from tableturf.ai.alpha.alpha import Alpha, Evaluation
from tableturf.model import DataLoader, CardFactory, StageFactory
from tableturf.model import Card, Stage, Status, Step, Grid

class EnhancedAlpha(Alpha):
    """增强版Alpha AI，支持JSON数据"""
    
    def __init__(self, npc_id: Optional[str] = None, level: int = 0):
        """
        初始化增强版Alpha AI
        
        Args:
            npc_id: 对战的NPC ID，如果提供则加载NPC数据
            level: NPC等级（0-2）
        """
        super().__init__()
        
        # 数据模块
        self.data_loader = DataLoader()
        self.card_factory = CardFactory(self.data_loader)
        self.stage_factory = StageFactory(self.data_loader)
        
        # NPC数据
        self.npc_id = npc_id
        self.npc_level = level
        self.npc_info = None
        self.npc_deck = None
        self.npc_stage = None
        
        # 加载NPC数据（如果提供）
        if npc_id:
            self.load_npc_data(npc_id, level)
    
    def load_npc_data(self, npc_id: str, level: int = 0):
        """加载NPC数据"""
        self.npc_id = npc_id
        self.npc_level = level
        
        # 加载NPC信息
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
            logger.warning("未加载NPC数据")
            return
        
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
    
    def _analyze_deck(self, deck: List[Card]) -> dict:
        """分析卡组特点"""
        if not deck:
            return {}
        
        # 卡牌面积分布
        sizes = [card.size for card in deck]
        sp_costs = [card.sp_cost for card in deck]
        
        # 特殊牌（有SP消耗的牌）
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
    
    def _analyze_stage(self, stage: Stage) -> dict:
        """分析场地特点"""
        return {
            "dimensions": stage.shape,
            "playable_cells": stage.size,
            "my_special_points": len(stage.my_sp),
            "his_special_points": len(stage.his_sp),
            "playable_percentage": (stage.size / (stage.shape[0] * stage.shape[1])) * 100
        }
    
    def redraw(self, hands: List[Card], stage: Optional[Stage] = None, 
               my_deck: Optional[List[Card]] = None, 
               his_deck: Optional[List[Card]] = None) -> bool:
        """
        重抽决策，增强版考虑NPC AI类型
        
        如果加载了NPC数据，会根据NPC的AI类型调整策略
        """
        # 如果有NPC数据，根据AI类型调整
        if self.npc_info and self.npc_level < len(self.npc_info.get("AIType", [])):
            ai_type = self.npc_info["AIType"][self.npc_level]
            
            # 根据AI类型调整重抽策略
            if ai_type == "Aggressive":
                # 激进型AI：更倾向于重抽以获取更好的手牌
                return self._aggressive_redraw(hands, stage, my_deck, his_deck)
            elif ai_type == "Defensive":
                # 防御型AI：更保守，不轻易重抽
                return self._defensive_redraw(hands, stage, my_deck, his_deck)
        
        # 默认使用原始Alpha算法
        return super().redraw(hands, stage, my_deck, his_deck)
    
    def _aggressive_redraw(self, hands: List[Card], stage: Optional[Stage] = None,
                          my_deck: Optional[List[Card]] = None,
                          his_deck: Optional[List[Card]] = None) -> bool:
        """激进型重抽策略"""
        # 激进策略：只要手牌平均面积小于8就重抽
        avg_size = sum(card.size for card in hands) / len(hands)
        return avg_size < 8
    
    def _defensive_redraw(self, hands: List[Card], stage: Optional[Stage] = None,
                         my_deck: Optional[List[Card]] = None,
                         his_deck: Optional[List[Card]] = None) -> bool:
        """防御型重抽策略"""
        # 防御策略：只有手牌特别差才重抽
        sizes = sorted([card.size for card in hands])
        # 如果最小的两张牌面积都小于4，才重抽
        return sizes[0] < 4 and sizes[1] < 4
    
    def next_step(self, status: Status) -> Step:
        """
        下一步决策，增强版考虑场地特性和NPC策略
        """
        # 如果有场地数据，分析场地特性
        if self.npc_stage:
            stage_analysis = self._analyze_stage(status.stage)
            
            # 根据场地特性调整策略
            if stage_analysis["playable_percentage"] < 50:
                # 小场地：采用更激进的策略
                logger.debug("小场地检测，采用激进策略")
                return self._aggressive_next_step(status)
            elif stage_analysis["my_special_points"] > stage_analysis["his_special_points"]:
                # 我方特殊点多：采用扩张策略
                logger.debug("我方特殊点优势，采用扩张策略")
                return self._expanding_next_step(status)
        
        # 默认使用原始Alpha算法
        return super().next_step(status)
    
    def _aggressive_next_step(self, status: Status) -> Step:
        """激进策略：优先攻击敌方墨迹"""
        # 获取所有可能步骤
        steps = status.get_possible_steps()
        if not steps:
            return status.get_possible_steps()[0]
        
        # 评分：优先攻击敌方墨迹
        scores = []
        for step in steps:
            if step.action == Step.Action.Skip:
                scores.append(-100)  # 跳过是最后的选择
                continue
            
            next_stage = util.estimate_stage(status.stage, step)
            
            # 计算攻击得分：减少的敌方墨迹 + 增加的敌方SP消耗
            his_ink_reduction = len(status.stage.his_ink) - len(next_stage.his_ink)
            his_sp_increase = len(next_stage.his_fiery_sp) - len(status.stage.his_fiery_sp)
            
            score = his_ink_reduction * 10 - his_sp_increase * 5
            
            # 考虑卡牌使用效率
            if step.action == Step.Action.Place:
                score += step.card.size * 2
            
            scores.append(score)
        
        return steps[np.argmax(scores)]
    
    def _expanding_next_step(self, status: Status) -> Step:
        """扩张策略：优先扩大领地"""
        # 获取所有放置步骤
        steps = status.get_possible_steps(action=Step.Action.Place)
        if not steps:
            return super().next_step(status)  # 回退到父类Alpha的默认策略（而非self.next_step避免无限递归）
        
        # 使用Alpha的扩张评分
        current_score = self._Alpha__score_current_stage(status)
        scores = np.array([self._Alpha__score_expanding_step(current_score, status, step) 
                          for step in steps])
        
        return steps[np.argmax(np.sum(scores, axis=1))]
    
    def get_recommended_counter_npc(self) -> List[tuple]:
        """获取推荐克制的NPC列表"""
        if not self.npc_info or not self.npc_deck:
            return []
        
        # 分析当前NPC的卡组
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
            
            # 克制评分：面积优势 + 特殊牌优势
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

# 使用示例
def example_usage():
    """使用示例"""
    # 创建增强版Alpha AI，针对特定NPC
    ai = EnhancedAlpha(npc_id="MiniGame_Aori", level=1)
    
    # 分析NPC策略
    analysis = ai.analyze_npc_strategy()
    print("NPC策略分析:")
    print(f"  NPC: {analysis['name']} ({analysis['name_zh']})")
    print(f"  AI类型: {analysis['ai_types']}")
    print(f"  AI等级: {analysis['ai_levels']}")
    
    if 'deck_analysis' in analysis:
        deck = analysis['deck_analysis']
        print(f"  卡组分析:")
        print(f"    卡牌数量: {deck['total_cards']}")
        print(f"    平均面积: {deck['avg_size']:.1f}")
        print(f"    特殊牌比例: {deck['special_cards_ratio']:.1%}")
    
    # 获取推荐克制的NPC
    counters = ai.get_recommended_counter_npc()
    print(f"\n推荐克制的NPC ({len(counters)} 个):")
    for i, (npc_id, score, name, name_zh, avg_size, special_ratio) in enumerate(counters[:3]):
        print(f"  {i+1}. {name} ({name_zh}): 评分={score:.2f}, 平均面积={avg_size:.1f}")

if __name__ == "__main__":
    example_usage()