#!/usr/bin/env python3
"""
AI模块数据集成示例
展示如何在AI中使用新的数据模块
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tableturf.model import DataLoader, CardFactory, StageFactory
from tableturf.model.status import Status
from tableturf.model.step import Step

class EnhancedAI:
    """增强版AI，使用JSON数据"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.card_factory = CardFactory(self.data_loader)
        self.stage_factory = StageFactory(self.data_loader)
        
        # 缓存加载的数据
        self.npc_cache = {}
        self.card_cache = {}
        self.stage_cache = {}
        
    def load_npc_data(self, npc_id: str, level: int = 0):
        """加载NPC相关数据"""
        cache_key = f"{npc_id}_level{level}"
        if cache_key in self.npc_cache:
            return self.npc_cache[cache_key]
        
        print(f"加载NPC数据: {npc_id} (等级 {level})")
        
        # 获取NPC信息
        npc_info = self.data_loader.get_npc_by_id(npc_id)
        if not npc_info:
            print(f"错误: 未找到NPC {npc_id}")
            return None, [], None
        
        # 获取NPC的卡组
        deck = self.card_factory.create_deck_from_npc_id(npc_id)
        
        # 获取场地
        stage = None
        map_list = npc_info.get("Map", [])
        if map_list and level < len(map_list):
            stage_name = map_list[level]
            stage = self.stage_factory.create_stage_from_name(stage_name)
        
        result = (npc_info, deck, stage)
        self.npc_cache[cache_key] = result
        return result
    
    def analyze_npc_strategy(self, npc_id: str):
        """分析NPC的策略特点"""
        npc_info, deck, stage = self.load_npc_data(npc_id)
        if not npc_info:
            return
        
        print(f"\nNPC策略分析: {npc_info.get('Name', '')} ({npc_info.get('name_zh', '')})")
        print("-" * 40)
        
        # AI类型分析
        ai_types = npc_info.get("AIType", [])
        ai_levels = npc_info.get("AILevel", [])
        print(f"AI配置: {list(zip(ai_levels, ai_types))}")
        
        # 卡组分析
        if deck:
            print(f"\n卡组分析 ({len(deck)} 张卡牌):")
            
            # 按面积分组
            size_groups = {}
            for card in deck:
                size = card.size
                size_groups[size] = size_groups.get(size, 0) + 1
            
            print("卡牌面积分布:")
            for size in sorted(size_groups.keys()):
                count = size_groups[size]
                print(f"  {size}格: {count}张 ({count/len(deck)*100:.1f}%)")
            
            # SP消耗分析
            sp_costs = [card.sp_cost for card in deck]
            avg_sp = sum(sp_costs) / len(sp_costs) if sp_costs else 0
            max_sp = max(sp_costs) if sp_costs else 0
            print(f"SP消耗: 平均 {avg_sp:.1f}, 最高 {max_sp}")
            
            # 特殊牌分析
            special_cards = [card for card in deck if card.sp_cost > 0]
            print(f"特殊牌: {len(special_cards)}张 ({len(special_cards)/len(deck)*100:.1f}%)")
        
        # 场地分析
        if stage:
            print(f"\n场地分析:")
            print(f"  尺寸: {stage.shape[0]}x{stage.shape[1]}")
            print(f"  可涂地格子: {stage.size}")
            print(f"  我方特殊点: {len(stage.my_sp)}")
            print(f"  敌方特殊点: {len(stage.his_sp)}")
    
    def create_simulation_status(self, npc_id: str, level: int = 0):
        """创建模拟对战状态"""
        npc_info, npc_deck, stage = self.load_npc_data(npc_id, level)
        if not npc_info or not stage:
            print(f"无法创建模拟状态: 缺少NPC数据或场地")
            return None
        
        print(f"创建模拟对战: NPC={npc_info.get('Name', 'Unknown')}, 等级={level}")
        
        # 创建初始状态
        # 注意: 这里需要根据实际游戏规则设置初始手牌、SP等
        # 这只是一个示例
        
        # 假设我方使用默认卡组（前15张卡牌）
        my_deck = npc_deck[:15] if len(npc_deck) >= 15 else npc_deck
        my_hands = my_deck[:4] if len(my_deck) >= 4 else my_deck
        
        # 假设敌方使用相同卡组
        his_deck = npc_deck[:15] if len(npc_deck) >= 15 else npc_deck
        
        # 创建Status对象
        # 注意: 这里简化了Status的创建，实际需要根据游戏规则
        try:
            status = Status(
                stage=stage,
                hands=my_hands,
                round=12,  # 初始回合数
                my_sp=0,   # 初始SP
                his_sp=0,  # 敌方初始SP
                my_deck=my_deck[4:] if len(my_deck) > 4 else [],  # 剩余牌组
                his_deck=his_deck  # 敌方牌组
            )
            
            print(f"模拟状态创建成功:")
            print(f"  回合数: {status.round}")
            print(f"  我方手牌: {len(status.hands)} 张")
            print(f"  我方牌组剩余: {len(status.my_deck)} 张")
            print(f"  敌方牌组: {len(status.his_deck)} 张")
            print(f"  场地: {stage.shape[0]}x{stage.shape[1]}")
            
            return status
        except Exception as e:
            print(f"创建Status对象失败: {e}")
            return None
    
    def find_counter_npc(self, target_npc_id: str):
        """寻找克制指定NPC的卡组策略"""
        target_info, target_deck, _ = self.load_npc_data(target_npc_id)
        if not target_info or not target_deck:
            return []
        
        print(f"\n寻找克制 {target_info.get('Name', 'Unknown')} 的策略...")
        
        # 分析目标NPC的卡组特点
        target_avg_size = sum(card.size for card in target_deck) / len(target_deck)
        target_special_ratio = len([c for c in target_deck if c.sp_cost > 0]) / len(target_deck)
        
        print(f"目标NPC卡组特点:")
        print(f"  平均面积: {target_avg_size:.1f}")
        print(f"  特殊牌比例: {target_special_ratio:.1%}")
        
        # 寻找所有NPC
        all_npc_ids = self.data_loader.get_all_npc_ids()
        counter_candidates = []
        
        for npc_id in all_npc_ids:
            if npc_id == target_npc_id:
                continue
                
            _, deck, _ = self.load_npc_data(npc_id)
            if not deck:
                continue
            
            # 计算卡组特点
            avg_size = sum(card.size for card in deck) / len(deck)
            special_ratio = len([c for c in deck if c.sp_cost > 0]) / len(deck)
            
            # 简单的克制逻辑：使用更大面积的卡牌对抗
            size_advantage = avg_size - target_avg_size
            
            # 特殊牌优势
            special_advantage = special_ratio - target_special_ratio
            
            # 综合评分
            score = size_advantage * 0.7 + special_advantage * 0.3
            
            if score > 0:  # 有优势
                counter_candidates.append((npc_id, score, avg_size, special_ratio))
        
        # 按评分排序
        counter_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n找到 {len(counter_candidates)} 个潜在克制NPC:")
        for i, (npc_id, score, avg_size, special_ratio) in enumerate(counter_candidates[:5]):
            npc_info = self.data_loader.get_npc_by_id(npc_id)
            name = npc_info.get('Name', 'Unknown') if npc_info else 'Unknown'
            name_zh = npc_info.get('name_zh', '') if npc_info else ''
            print(f"{i+1}. {name} ({name_zh}): 评分={score:.2f}, 平均面积={avg_size:.1f}, 特殊牌比例={special_ratio:.1%}")
        
        return counter_candidates

def main():
    """主函数"""
    print("AI数据集成示例")
    print("=" * 60)
    
    ai = EnhancedAI()
    
    # 获取所有NPC
    loader = DataLoader()
    npc_ids = loader.get_all_npc_ids()
    
    if not npc_ids:
        print("未找到NPC数据")
        return
    
    print(f"找到 {len(npc_ids)} 个NPC")
    
    # 示例1: 分析第一个NPC
    sample_npc = npc_ids[0]
    print(f"\n示例1: 分析NPC '{sample_npc}'")
    ai.analyze_npc_strategy(sample_npc)
    
    # 示例2: 创建模拟对战
    print(f"\n示例2: 创建模拟对战")
    status = ai.create_simulation_status(sample_npc, level=0)
    
    if status:
        # 显示可能的行动
        possible_steps = status.get_possible_steps()
        print(f"当前可能的行动数: {len(possible_steps)}")
        
        if possible_steps:
            print("前3个可能行动:")
            for i, step in enumerate(possible_steps[:3]):
                print(f"  {i+1}. {step}")
    
    # 示例3: 寻找克制策略
    if len(npc_ids) > 1:
        print(f"\n示例3: 寻找克制策略")
        target_npc = npc_ids[1] if len(npc_ids) > 1 else npc_ids[0]
        ai.find_counter_npc(target_npc)
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)

if __name__ == "__main__":
    main()