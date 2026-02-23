# 数据模块使用指南

## 概述

本文档介绍了新创建的数据管理模块，这些模块用于加载和管理Tableturf游戏的JSON数据文件。

## 模块结构

### 1. DataLoader (`data_loader.py`)
**功能**: 加载和管理所有JSON数据文件
**特点**: 单例模式，自动缓存，懒加载

### 2. CardFactory (`card_factory.py`)
**功能**: 根据JSON数据创建Card对象
**特点**: 数据验证，错误处理，缓存支持

### 3. StageFactory (`stage_factory.py`)
**功能**: 根据JSON数据创建Stage对象
**特点**: 数据验证，场地分析，缓存支持

## 快速开始

### 基本使用

```python
from tableturf.model import DataLoader, CardFactory, StageFactory

# 创建实例
loader = DataLoader()  # 单例模式，多次调用返回同一实例
card_factory = CardFactory(loader)
stage_factory = StageFactory(loader)

# 加载所有数据（可选，懒加载会在需要时自动加载）
loader.load_all()

# 获取NPC信息
npc_info = loader.get_npc_by_id("MiniGame_Aori")
print(f"NPC: {npc_info['Name']} ({npc_info['name_zh']})")

# 创建NPC卡组
deck = card_factory.create_deck_from_npc_id("MiniGame_Aori")
print(f"卡组大小: {len(deck)} 张卡牌")

# 创建场地
stage = stage_factory.create_stage_from_name("Square")
print(f"场地尺寸: {stage.shape}")
```

### 数据关系示例

```python
# 获取NPC的完整数据
def get_npc_complete_data(npc_id, level=0):
    """获取NPC的完整数据"""
    loader = DataLoader()
    
    # 1. NPC基本信息
    npc_info = loader.get_npc_by_id(npc_id)
    
    # 2. NPC卡组
    card_refs = loader.get_deck_by_npc_id(npc_id)
    card_infos = [loader.get_card_by_id(ref) for ref in card_refs]
    
    # 3. NPC场地
    map_list = npc_info.get("Map", [])
    stage_name = map_list[level] if level < len(map_list) else None
    stage_info = loader.get_stage_by_name(stage_name) if stage_name else None
    
    return {
        "npc": npc_info,
        "deck_cards": card_infos,
        "stage": stage_info
    }
```

## API参考

### DataLoader 类

#### 主要方法

| 方法 | 描述 | 返回值 |
|------|------|--------|
| `load_all()` | 加载所有JSON数据 | None |
| `get_card_by_id(card_id)` | 根据ID获取卡牌信息 | dict 或 None |
| `get_npc_by_id(npc_id)` | 根据ID获取NPC信息 | dict 或 None |
| `get_deck_by_npc_id(npc_id)` | 获取NPC的卡组引用 | list |
| `get_stage_by_name(name)` | 根据名称获取场地信息 | dict 或 None |
| `get_all_npc_ids()` | 获取所有NPC ID | list |
| `get_all_card_ids()` | 获取所有卡牌ID | list |
| `get_all_stage_names()` | 获取所有场地名称 | list |
| `get_npc_deck_cards(npc_id)` | 获取NPC卡组的卡牌信息 | list |
| `get_npc_stage_info(npc_id, level)` | 获取NPC的场地信息 | dict 或 None |
| `clear_cache()` | 清除缓存 | None |

### CardFactory 类

#### 主要方法

| 方法 | 描述 | 返回值 |
|------|------|--------|
| `create_card_from_id(card_id)` | 根据ID创建Card对象 | Card 或 None |
| `create_deck_from_npc_id(npc_id)` | 创建NPC的卡组 | list[Card] |
| `get_card_info(card_id)` | 获取卡牌信息 | dict 或 None |
| `get_card_properties(card_id)` | 获取卡牌属性摘要 | dict |
| `print_card_summary(card_id)` | 打印卡牌摘要 | None |

### StageFactory 类

#### 主要方法

| 方法 | 描述 | 返回值 |
|------|------|--------|
| `create_stage_from_name(name)` | 根据名称创建Stage对象 | Stage 或 None |
| `get_stage_info(name)` | 获取场地信息 | dict 或 None |
| `get_stage_properties(name)` | 获取场地属性摘要 | dict |
| `analyze_stage(name)` | 分析场地特性 | dict |
| `print_stage_summary(name)` | 打印场地摘要 | None |

## 在AI模块中使用

### 示例：增强版AI

```python
from tableturf.model import DataLoader, CardFactory, StageFactory
from tableturf.ai import AI

class EnhancedAI(AI):
    def __init__(self):
        super().__init__()
        self.data_loader = DataLoader()
        self.card_factory = CardFactory(self.data_loader)
        self.stage_factory = StageFactory(self.data_loader)
        
    def redraw(self, hands, stage=None, my_deck=None, his_deck=None):
        # 使用JSON数据增强决策
        npc_info = self.data_loader.get_npc_by_id("current_npc_id")
        ai_type = npc_info.get("AIType", ["Normal"])[0]
        
        # 根据AI类型调整策略
        if ai_type == "Aggressive":
            return self._aggressive_redraw(hands, stage)
        elif ai_type == "Defensive":
            return self._defensive_redraw(hands, stage)
        else:
            return super().redraw(hands, stage, my_deck, his_deck)
    
    def next_step(self, status):
        # 使用场地信息优化决策
        stage_analysis = self.stage_factory.analyze_stage("current_stage")
        
        # 根据场地特性调整策略
        if stage_analysis.get("playable_percentage", 0) < 50:
            # 小场地，采用激进策略
            return self._aggressive_step(status)
        else:
            # 大场地，采用保守策略
            return self._conservative_step(status)
```

### 示例：NPC策略分析

```python
class NPCAnalyzer:
    def __init__(self):
        self.loader = DataLoader()
        self.card_factory = CardFactory(self.loader)
        
    def analyze_npc_deck(self, npc_id):
        """分析NPC卡组策略"""
        deck = self.card_factory.create_deck_from_npc_id(npc_id)
        
        analysis = {
            "total_cards": len(deck),
            "avg_size": sum(card.size for card in deck) / len(deck),
            "total_sp_cost": sum(card.sp_cost for card in deck),
            "special_cards": len([c for c in deck if c.sp_cost > 0]),
            "size_distribution": self._get_size_distribution(deck)
        }
        
        return analysis
    
    def find_counter_deck(self, target_npc_id):
        """寻找克制指定NPC的卡组"""
        target_deck = self.card_factory.create_deck_from_npc_id(target_npc_id)
        target_avg_size = sum(card.size for card in target_deck) / len(target_deck)
        
        counter_candidates = []
        for npc_id in self.loader.get_all_npc_ids():
            if npc_id == target_npc_id:
                continue
                
            deck = self.card_factory.create_deck_from_npc_id(npc_id)
            avg_size = sum(card.size for card in deck) / len(deck)
            
            # 简单克制逻辑：使用更大面积的卡牌
            if avg_size > target_avg_size * 1.2:
                counter_candidates.append((npc_id, avg_size))
        
        return sorted(counter_candidates, key=lambda x: x[1], reverse=True)
```

## 数据文件说明

### 1. MiniGameCardInfo_Matrix_ZH.json
- **描述**: 所有卡牌信息
- **关键字段**:
  - `__RowId`: 卡牌唯一标识
  - `square_matrix`: 8x8卡牌矩阵
  - `SpecialCost`: SP消耗
  - `area`: 卡牌面积
  - `name_zh`: 中文名称

### 2. MiniGameGameNpcData_ZH.json
- **描述**: NPC信息
- **关键字段**:
  - `__RowId`: NPC唯一标识
  - `Deck`: 卡组引用列表
  - `Map`: 场地名称列表
  - `AILevel`: AI等级列表
  - `AIType`: AI类型列表

### 3. MiniGamePresetDeck.json
- **描述**: 预设卡组
- **关键字段**:
  - `__RowId`: 卡组标识（与NPC ID对应）
  - `Card`: 卡牌引用列表

### 4. MiniGameStage.json
- **描述**: 场地信息
- **关键字段**:
  - `Name`: 场地名称
  - `square_matrix`: 场地矩阵
  - `Dimensions`: 场地尺寸
  - `SpecialPoints`: 特殊点列表

## 最佳实践

### 1. 使用单例模式
```python
# 正确：使用单例
loader1 = DataLoader()
loader2 = DataLoader()
print(loader1 is loader2)  # True，同一实例

# 避免：创建多个实例浪费内存
```

### 2. 懒加载优化
```python
# 数据在第一次访问时自动加载，无需手动调用load_all()
card_info = loader.get_card_by_id("MiniGame_Aori")  # 自动加载
```

### 3. 错误处理
```python
try:
    card = card_factory.create_card_from_id(card_id)
    if card is None:
        print(f"警告：卡牌 {card_id} 创建失败")
        # 使用默认卡牌或跳过
except Exception as e:
    print(f"错误：{e}")
    # 适当的错误处理
```

### 4. 缓存利用
```python
# 频繁访问的数据会被缓存
for i in range(10):
    card = card_factory.create_card_from_id("MiniGame_Aori")  # 只加载一次
    
# 清除缓存（当数据更新时）
loader.clear_cache()
```

## 测试脚本

项目包含以下测试脚本：

1. `simple_test.py` - 基本JSON文件加载测试
2. `test_data_modules.py` - 完整模块测试（需要numpy）
3. `data_integration_example.py` - AI集成示例

运行测试：
```bash
# 基本测试
python3 tableturf/model/simple_test.py

# 创建测试文件后运行完整测试
python3 tableturf/model/test_data_modules.py
```

## 故障排除

### 常见问题

1. **JSON解析错误**
   - 检查文件编码（应为UTF-8）
   - 检查JSON格式（使用json.tool验证）
   - 修复制表符和尾随逗号

2. **模块导入错误**
   - 确保Python路径包含项目根目录
   - 检查依赖是否安装（numpy, opencv-python）

3. **数据关系错误**
   - 验证NPC ID与卡组ID的对应关系
   - 检查场地名称是否匹配

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据加载
loader = DataLoader()
print(f"NPC数量: {len(loader.get_all_npc_ids())}")
print(f"卡牌数量: {len(loader.get_all_card_ids())}")
print(f"场地数量: {len(loader.get_all_stage_names())}")

# 验证数据关系
npc_id = "MiniGame_Aori"
deck_refs = loader.get_deck_by_npc_id(npc_id)
print(f"NPC {npc_id} 的卡组引用: {len(deck_refs)} 个")
```

## 扩展开发

### 添加新数据源

1. 在`DataLoader`中添加新的加载方法
2. 创建对应的工厂类（如果需要创建对象）
3. 更新`__init__.py`导出新类

### 自定义数据处理

```python
class CustomCardFactory(CardFactory):
    def create_card_with_custom_rules(self, card_id, custom_rules):
        """使用自定义规则创建卡牌"""
        card_info = self.get_card_info(card_id)
        if not card_info:
            return None
        
        # 应用自定义规则
        modified_matrix = self._apply_custom_rules(
            card_info['square_matrix'], 
            custom_rules
        )
        
        return self.create_card_from_info({
            **card_info,
            'square_matrix': modified_matrix
        })
```

## 总结

新的数据模块提供了：
- 统一的数据加载和管理
- 类型安全的对象创建
- 高效的数据缓存
- 灵活的数据关系查询
- 易于集成的API设计

这些模块可以显著简化AI开发，提供更丰富的游戏数据支持。