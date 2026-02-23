import json
import os
from typing import Dict, List, Any, Optional
import numpy as np

class DataLoader:
    """数据加载器，负责加载和管理JSON数据"""
    
    _instance = None
    
    def __new__(cls, data_dir: str = None):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, data_dir: str = None):
        if self._initialized:
            return
            
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.card_info: Optional[Dict[str, Dict]] = None
        self.npc_data: Optional[Dict[str, Dict]] = None
        self.preset_decks: Optional[Dict[str, Dict]] = None
        self.stages: Optional[Dict[str, Dict]] = None
        
        self._initialized = True
        
    def load_all(self):
        """加载所有JSON数据"""
        self.load_card_info()
        self.load_npc_data()
        self.load_preset_decks()
        self.load_stages()
        
    def load_card_info(self) -> Dict[str, Dict]:
        """加载卡牌信息"""
        path = os.path.join(self.data_dir, "MiniGameCardInfo_Matrix_ZH.json")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 转换为以__RowId为键的字典
            self.card_info = {item["__RowId"]: item for item in data}
            print(f"Loaded {len(self.card_info)} card info entries")
            return self.card_info
        except Exception as e:
            print(f"Error loading card info: {e}")
            self.card_info = {}
            return {}
    
    def load_npc_data(self) -> Dict[str, Dict]:
        """加载NPC数据"""
        path = os.path.join(self.data_dir, "MiniGameGameNpcData_ZH.json")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.npc_data = {item["__RowId"]: item for item in data}
            print(f"Loaded {len(self.npc_data)} NPC data entries")
            return self.npc_data
        except Exception as e:
            print(f"Error loading NPC data: {e}")
            self.npc_data = {}
            return {}
    
    def load_preset_decks(self) -> Dict[str, Dict]:
        """加载预设卡组"""
        path = os.path.join(self.data_dir, "MiniGamePresetDeck.json")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.preset_decks = {item["__RowId"]: item for item in data}
            print(f"Loaded {len(self.preset_decks)} preset deck entries")
            return self.preset_decks
        except Exception as e:
            print(f"Error loading preset decks: {e}")
            self.preset_decks = {}
            return {}
    
    def load_stages(self) -> Dict[str, Dict]:
        """加载场地信息"""
        path = os.path.join(self.data_dir, "MiniGameStage.json")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.stages = {item["Name"]: item for item in data}
            print(f"Loaded {len(self.stages)} stage entries")
            return self.stages
        except Exception as e:
            print(f"Error loading stages: {e}")
            self.stages = {}
            return {}
    
    def get_card_by_id(self, card_id: str) -> Optional[Dict]:
        """根据卡牌ID获取卡牌信息"""
        if self.card_info is None:
            self.load_card_info()
        # 处理引用格式：Work/Gyml/MiniGame_Aori.spl__MiniGameCardInfo.gyml
        if card_id.startswith("Work/Gyml/"):
            # 提取卡牌ID：MiniGame_Aori
            card_id = card_id.split("/")[-1].split(".")[0]
        return self.card_info.get(card_id)
    
    def get_npc_by_id(self, npc_id: str) -> Optional[Dict]:
        """根据NPC ID获取NPC信息"""
        if self.npc_data is None:
            self.load_npc_data()
        return self.npc_data.get(npc_id)
    
    def get_deck_by_npc_id(self, npc_id: str) -> List[str]:
        """根据NPC ID获取卡组引用列表"""
        if self.preset_decks is None:
            self.load_preset_decks()
        deck_data = self.preset_decks.get(npc_id)
        if deck_data:
            return deck_data.get("Card", [])
        return []
    
    def get_stage_by_name(self, stage_name: str) -> Optional[Dict]:
        """根据场地名称获取场地信息"""
        if self.stages is None:
            self.load_stages()
        return self.stages.get(stage_name)
    
    def convert_card_matrix(self, matrix: List[List[int]]) -> np.ndarray:
        """将JSON中的卡牌矩阵转换为numpy数组"""
        return np.array(matrix, dtype=np.int32)
    
    def convert_stage_matrix(self, matrix: List[List[int]]) -> np.ndarray:
        """将JSON中的场地矩阵转换为numpy数组"""
        return np.array(matrix, dtype=np.int32)
    
    def get_all_npc_ids(self) -> List[str]:
        """获取所有NPC ID"""
        if self.npc_data is None:
            self.load_npc_data()
        return list(self.npc_data.keys())
    
    def get_all_card_ids(self) -> List[str]:
        """获取所有卡牌ID"""
        if self.card_info is None:
            self.load_card_info()
        return list(self.card_info.keys())
    
    def get_all_stage_names(self) -> List[str]:
        """获取所有场地名称"""
        if self.stages is None:
            self.load_stages()
        return list(self.stages.keys())
    
    def get_npc_deck_cards(self, npc_id: str) -> List[Dict]:
        """根据NPC ID获取卡组中的卡牌信息列表"""
        card_refs = self.get_deck_by_npc_id(npc_id)
        cards = []
        for card_ref in card_refs:
            card_info = self.get_card_by_id(card_ref)
            if card_info:
                cards.append(card_info)
        return cards
    
    def get_npc_stage_info(self, npc_id: str, level: int = 0) -> Optional[Dict]:
        """根据NPC ID和等级获取场地信息"""
        npc_info = self.get_npc_by_id(npc_id)
        if not npc_info:
            return None
        
        # 获取场地名称（根据等级）
        map_list = npc_info.get("Map", [])
        if not map_list:
            return None
        
        # 确保等级在有效范围内
        level = max(0, min(level, len(map_list) - 1))
        stage_name = map_list[level]
        
        return self.get_stage_by_name(stage_name)
    
    def clear_cache(self):
        """清除缓存，强制重新加载数据"""
        self.card_info = None
        self.npc_data = None
        self.preset_decks = None
        self.stages = None
        print("Data cache cleared")