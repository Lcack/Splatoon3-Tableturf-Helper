# Autotableturf 项目开发总结

## 项目概述
Autotableturf 是一个自动化玩 Splatoon 3 Tableturf Battle 的游戏助手，通过图像识别和蓝牙模拟实现自动对战。

## 2026-02-23 主要开发工作

### 1. JSON 数据模块集成与验证
**目标**: 将游戏数据从JSON文件集成到AI系统中，确保数据一致性
**完成工作**:
- 创建 `DataLoader` 类：单例模式加载和管理4个JSON数据文件
- 创建 `CardFactory` 类：根据JSON数据创建Card对象，处理数值转换（0→Empty, 1→MyInk, 2→MySpecial）
- 创建 `StageFactory` 类：根据JSON数据创建Stage对象，处理数值转换（1→Empty, 4→MySpecial, 16→HisSpecial, 64→Wall）
- 修复 `MiniGameStage.json` 文件格式错误（制表符问题）
- 验证数据关系：NPC ↔ 卡组 ↔ 卡牌 ↔ 场地

**数据文件**:
- `MiniGameCardInfo_Matrix_ZH.json` (266张卡牌，矩阵值: 0-空白, 1-普通墨迹, 2-特殊墨迹)
- `MiniGameGameNpcData_ZH.json` (30个NPC)
- `MiniGamePresetDeck.json` (31个预设卡组)
- `MiniGameStage.json` (15个场地，矩阵值: 1-可涂地, 4-我方特殊点, 16-敌方特殊点, 64-墙壁)

### 2. MCTS AI 算法实现与修复
**目标**: 实现蒙特卡洛树搜索算法并修复现有代码问题
**完成工作**:
- 分析原始MCTS代码问题：调用了不存在的方法，与游戏规则不匹配
- 创建 `MCTSAI` 类：符合Tableturf游戏规则的MCTS实现
- 创建 `EnhancedMCTSAI` 类：集成JSON数据的增强版MCTS
- 实现完整MCTS流程：选择→扩展→模拟→回溯
- 添加启发式引导和状态评估函数

**修复的问题**:
- 缺失方法：`is_terminal()`, `simulate_step()`, `apply_step()`, `preview_step()`
- 游戏规则：正确处理12回合、重抽、特殊攻击、SP点数
- 性能优化：时间限制、模拟次数限制、启发式快速走子

### 3. 增强版Alpha AI
**目标**: 基于JSON数据增强现有Alpha算法
**完成工作**:
- 创建 `EnhancedAlpha` 类：集成JSON数据支持的Alpha AI
- 添加NPC策略分析功能
- 实现克制NPC推荐系统
- 根据NPC AI类型自适应调整策略

### 4. 检测模块重构与增强（JSON数据验证）
**目标**: 重构检测模块，集成JSON数据验证，提高识别准确性
**完成工作**:
- **重构card检测**: 创建 `enhanced_card.py` 完全替代原 `card.py`
  - 实现 `EnhancedCardDetector` 类：结合图像识别和JSON数据验证
  - 卡牌匹配算法：SP消耗(40%) + 卡牌大小(30%) + 图案形状(30%)
  - 匹配阈值：70%匹配度，低于阈值使用基础检测
  - **完全上位替代**: `enhanced_card.hands()` 函数完全兼容原接口
  - **删除原文件**: 移除 `tableturf/manager/detection/card.py`

- **增强stage检测**: 创建 `enhanced_stage.py` 提供JSON验证
  - 实现 `EnhancedStageDetector` 类：集成场地数据验证
  - 场地匹配算法：尺寸(40%) + 可涂地比例(30%) + 特殊点数量(30%)
  - 匹配阈值：60%匹配度

- **更新模块导出**: 修改 `detection/__init__.py` 使用增强版函数

### 5. 线性学习MCTS AI（无深度学习依赖）
**目标**: 为MCTS添加学习能力，不依赖PyTorch等深度学习框架
**完成工作**:
- 创建 `LinearLearner` 类：简单的线性回归学习器
- 创建 `LearningMCTSAI` 类：带线性学习的MCTS AI
- 实现特征提取：墨迹数量、SP点数、特殊点数量、回合数、手牌特征等10维特征
- 添加时间差分学习：从模拟路径中学习状态价值
- 实现模型保存/加载：使用pickle保存学习权重
- 添加对战分析：记录胜率、决策时间、特征重要性等统计

**学习特性**:
- 梯度下降更新权重，学习率0.01
- 批量训练（每50个样本）
- 特征重要性分析
- 模型自动保存（每200个训练样本）

### 6. 代码清理和优化
**完成工作**:
- 删除所有 `Zone.Identifier` 冗余文件
- 修复空的 `__init__.py` 文件（特别是tableturf/__init__.py）
- 统一模块导入和导出
- 更新AI模块的 `__all__` 导出列表
- 清理测试文件，只保留核心代码
- **文件结构调整**:
  - 保留 `tableturf/model/card.py` (核心Card类)
  - 保留 `tableturf/model/card_factory.py` (Card工厂)
  - 保留 `tableturf/model/stage.py` (核心Stage类)
  - 保留 `tableturf/model/stage_factory.py` (Stage工厂)
  - 删除 `tableturf/manager/detection/card.py` (被enhanced_card.py替代)

### 7. 环境验证
**完成工作**:
- 验证conda环境 `autotableturf` 状态良好
- 确认所有依赖包已安装：numpy, opencv-python, requests, pyserial, flask等
- 运行完整测试验证所有新功能

## 新增模块结构

```
tableturf/
├── model/
│   ├── data_loader.py      # JSON数据加载器
│   ├── card_factory.py     # 卡牌对象工厂（处理数值转换）
│   ├── stage_factory.py    # 场地对象工厂（处理数值转换）
│   └── DATA_MODULES_GUIDE.md # 数据模块使用指南
├── ai/
│   ├── alpha/
│   │   ├── enhanced_alpha.py  # 增强版Alpha AI
│   │   └── Alpha算法分析.md   # Alpha算法文档
│   └── mcts/
│       ├── fixed_mcts.py      # 修复后的MCTS实现
│       ├── learning_mcts_complete.py  # 带线性学习的MCTS
│       └── tableturf_rule.md  # 游戏规则文档
├── manager/detection/
│   ├── enhanced_card.py    # 增强版卡牌检测（JSON验证）
│   └── __init__.py         # 导出增强版检测函数
```

## 可用AI类

1. **SimpleAI**: 随机选择行动
2. **Alpha**: 基于规则的启发式算法
3. **EnhancedAlpha**: 集成JSON数据的增强版Alpha
4. **MCTSAI**: 蒙特卡洛树搜索算法
5. **EnhancedMCTSAI**: 集成JSON数据的增强版MCTS
6. **LearningMCTSAI**: 带线性学习的MCTS AI（推荐）

## 使用示例

```python
# 基本MCTS AI
from tableturf.ai import MCTSAI
ai = MCTSAI(simulation_limit=200, time_limit_ms=1000)

# 增强版MCTS AI（集成JSON数据）
from tableturf.ai import EnhancedMCTSAI
ai = EnhancedMCTSAI(npc_id="MiniGame_Aori", level=1)

# 学习版MCTS AI（推荐）
from tableturf.ai.mcts.learning_mcts_complete import LearningMCTSAI
ai = LearningMCTSAI(simulation_limit=200, learning_enabled=True)

# 增强版卡牌检测
from tableturf.manager.detection import enhanced_hands
from tableturf.model import DataLoader, CardFactory

loader = DataLoader()
factory = CardFactory(loader)
cards = enhanced_hands(image, data_loader=loader, card_factory=factory)
```

## 技术要点

### 1. 数据一致性处理
- **卡牌JSON**: 0→Empty(1), 1→MyInk(2), 2→MySpecial(4)
- **场地JSON**: 1→Empty(1), 4→MySpecial(4), 16→HisSpecial(16), 64→Wall(64)
- **Factory类自动处理转换**，确保与Grid枚举一致

### 2. 检测增强
- **图像识别 + JSON验证** 双保险
- **匹配算法**: SP消耗(40%) + 卡牌大小(30%) + 图案形状(30%)
- **向后兼容**: 无JSON数据时自动降级为纯图像识别

### 3. 线性学习（无深度学习依赖）
- **特征工程**: 10维游戏状态特征
- **梯度下降**: 简单有效的权重更新
- **时间差分学习**: 从模拟路径中学习
- **模型持久化**: pickle保存，无需训练即可使用

### 4. 性能优化
- **MCTS参数可调**: 模拟次数、时间限制、探索权重
- **学习可控**: 可启用/禁用学习，调整学习率
- **内存管理**: 限制训练数据和历史记录大小

## 后续建议

1. **性能优化**: 添加MCTS并行化，缓存优化
2. **学习增强**: 添加更多特征，优化学习算法
3. **用户界面**: 在Web界面中添加AI选择和参数调整
4. **对战分析**: 添加对战记录和策略分析功能
5. **模型共享**: 创建预训练模型库

## 环境状态
✅ Conda环境配置完成  
✅ 所有依赖包已安装  
✅ 新模块测试通过  
✅ 代码清理完成  
✅ 文档更新完成  
✅ GitHub推送完成  

## 2026-02-23 文件结构调整总结

### 关键决策和调整

#### 1. **model目录文件必要性分析**
✅ **确认所有文件必须保留**：
- `card.py` / `stage.py`：**核心数据模型**，定义游戏对象结构和行为
- `card_factory.py` / `stage_factory.py`：**工厂模式**，负责从JSON数据创建对象
- **关系**：Factory依赖Model，这是标准的关注点分离设计

#### 2. **detection目录重构**
✅ **完成文件优化**：
- **删除** `tableturf/manager/detection/card.py`：被增强版完全替代
- **创建** `enhanced_card.py`：**完全上位替代**，提供：
  - 相同的`hands()`函数接口
  - 图像识别 + JSON数据验证双保险
  - 智能降级（JSON不可用时使用基础检测）
- **创建** `enhanced_stage.py`：提供场地JSON验证（可选增强）
- **更新** `__init__.py`：导出增强版函数

#### 3. **无重复代码**
✅ **检查确认**：
- 生产代码中没有硬编码的Card/Stage创建
- 测试文件中的硬编码是正常的单元测试需要
- Factory模式已正确实现，新代码应使用Factory

### 当前项目架构

```
架构层次：
1. 数据层 (model/)        : Card/Stage定义 + Factory创建 + JSON数据
2. 检测层 (detection/)    : 图像识别 + JSON验证（增强版）
3. AI层 (ai/)            : 多种AI算法 + 学习能力
4. 管理层 (manager/)      : 游戏流程控制
```

### 技术优势总结

1. **数据一致性**: Factory类确保JSON到Grid枚举的正确转换
2. **检测准确性**: 增强版检测提供双保险验证
3. **学习能力**: 线性学习MCTS无需深度学习依赖
4. **向后兼容**: 所有新功能与现有代码兼容
5. **可维护性**: 清晰的架构分层和关注点分离

项目已全面升级，现在具备数据驱动决策、学习能力和增强检测功能！