import numpy as np
from typing import List, Optional
from tableturf.model.card import Card
from tableturf.model.grid import Grid

class CardFactory:
    """卡牌工厂，根据JSON数据创建Card对象"""
    
    def __init__(self, data_loader=None):
        from tableturf.model.data_loader import DataLoader
        self.data_loader = data_loader or DataLoader()
        
    def create_card_from_id(self, card_id: str) -> Optional[Card]:
        """根据卡牌ID创建Card对象"""
        card_info = self.data_loader.get_card_by_id(card_id)
        if not card_info:
            print(f"Warning: Card not found: {card_id}")
            return None
        
        # 获取卡牌矩阵
        matrix = card_info.get("square_matrix", [])
        if not matrix:
            print(f"Warning: Card matrix not found: {card_id}")
            return None
        
        # 转换为numpy数组
        try:
            grid = self.data_loader.convert_card_matrix(matrix)
        except Exception as e:
            print(f"Error converting card matrix for {card_id}: {e}")
            return None
        
        # 转换矩阵值：根据Tableturf游戏逻辑
        # 在Tableturf中，卡牌矩阵通常使用以下值：
        # 0: 空白，1: 普通墨迹，2: 特殊墨迹（SP点）
        converted_grid = np.zeros_like(grid, dtype=int)
        
        # 将1转换为MyInk（普通墨迹）
        converted_grid[grid == 1] = Grid.MyInk.value
        
        # 将2转换为MySpecial（特殊墨迹，SP点）
        converted_grid[grid == 2] = Grid.MySpecial.value
        
        # 获取SP消耗
        sp_cost = card_info.get("SpecialCost", 0)
        
        # 获取卡牌面积（用于验证）
        area = card_info.get("area", 0)
        
        # 验证卡牌数据
        ink_count = np.sum(converted_grid == Grid.MyInk.value)
        special_count = np.sum(converted_grid == Grid.MySpecial.value)
        total_count = ink_count + special_count
        
        if area > 0 and total_count != area:
            print(f"Warning: Card area mismatch for {card_id}: expected {area}, got {total_count}")
        
        # 创建Card对象
        try:
            card = Card(converted_grid, sp_cost)
            return card
        except Exception as e:
            print(f"Error creating Card object for {card_id}: {e}")
            return None
    
    def create_deck_from_npc_id(self, npc_id: str) -> List[Card]:
        """根据NPC ID创建卡组（Card对象列表）"""
        card_refs = self.data_loader.get_deck_by_npc_id(npc_id)
        cards = []
        
        for i, card_ref in enumerate(card_refs):
            card = self.create_card_from_id(card_ref)
            if card:
                cards.append(card)
            else:
                print(f"Warning: Failed to create card {i+1} for NPC {npc_id}: {card_ref}")
        
        print(f"Created deck for NPC {npc_id}: {len(cards)} cards")
        return cards
    
    def create_card_from_info(self, card_info: dict) -> Optional[Card]:
        """直接从卡牌信息字典创建Card对象"""
        if not card_info:
            return None
        
        # 获取卡牌矩阵
        matrix = card_info.get("square_matrix", [])
        if not matrix:
            return None
        
        # 转换为numpy数组
        try:
            grid = self.data_loader.convert_card_matrix(matrix)
        except Exception as e:
            print(f"Error converting card matrix: {e}")
            return None
        
        # 转换矩阵值
        converted_grid = np.zeros_like(grid, dtype=int)
        converted_grid[grid == 1] = Grid.MyInk.value
        converted_grid[grid == 2] = Grid.MySpecial.value
        
        # 获取SP消耗
        sp_cost = card_info.get("SpecialCost", 0)
        
        # 创建Card对象
        try:
            return Card(converted_grid, sp_cost)
        except Exception as e:
            print(f"Error creating Card object: {e}")
            return None
    
    def get_card_info(self, card_id: str) -> Optional[dict]:
        """获取卡牌信息（不创建Card对象）"""
        return self.data_loader.get_card_by_id(card_id)
    
    def get_card_properties(self, card_id: str) -> dict:
        """获取卡牌属性摘要"""
        card_info = self.get_card_info(card_id)
        if not card_info:
            return {}
        
        return {
            "id": card_id,
            "name": card_info.get("Name", ""),
            "name_zh": card_info.get("name_zh", ""),
            "area": card_info.get("area", 0),
            "sp_cost": card_info.get("SpecialCost", 0),
            "rarity": card_info.get("Rarity", ""),
            "category": card_info.get("Category", ""),
            "season": card_info.get("Season", 0)
        }
    
    def validate_card_matrix(self, matrix: List[List[int]]) -> bool:
        """验证卡牌矩阵是否有效"""
        if not matrix:
            return False
        
        # 检查矩阵维度
        if len(matrix) != 8 or any(len(row) != 8 for row in matrix):
            print("Invalid matrix dimensions: expected 8x8")
            return False
        
        # 检查矩阵值
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val not in [0, 1, 2]:
                    print(f"Invalid value at ({i},{j}): {val}")
                    return False
        
        return True
    
    def print_card_summary(self, card_id: str):
        """打印卡牌摘要信息"""
        card_info = self.get_card_info(card_id)
        if not card_info:
            print(f"Card not found: {card_id}")
            return
        
        props = self.get_card_properties(card_id)
        print(f"Card ID: {props['id']}")
        print(f"Name: {props['name']} ({props['name_zh']})")
        print(f"Area: {props['area']}")
        print(f"SP Cost: {props['sp_cost']}")
        print(f"Rarity: {props['rarity']}")
        print(f"Category: {props['category']}")
        print(f"Season: {props['season']}")
        
        # 显示矩阵预览
        matrix = card_info.get("square_matrix", [])
        if matrix:
            print("Matrix preview (first 4 rows):")
            for i in range(min(4, len(matrix))):
                row_preview = " ".join(str(x) for x in matrix[i][:8])
                print(f"  {row_preview}")