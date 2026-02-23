# Alpha 算法分析文档

## 1. 算法概述

Alpha 算法是 Autotableturf 项目中的一个智能 AI 算法，专门用于玩 Splatoon 3 中的 Tableturf Battle（桌面涂地战）游戏。它是一个基于规则和启发式评估的算法，通过多维度评分系统来决定每一步的最佳行动。

## 2. 核心功能实现

### a) 重抽决策 (`redraw` 方法)
- 根据手牌大小决定是否重抽起始手牌
- 使用分位数比较：比较手牌的第3大牌（hand_quantile=2）与牌组的第9大牌（deck_quantile=8）
- 特殊牌（12格大小，3点SP消耗）有特殊处理逻辑

### b) 回合策略 (`next_step` 方法)
算法根据当前回合数采用不同的策略：

#### **第1回合策略：**
- 最大化己方墨迹面积，最小化对手墨迹面积
- 使用简单的评分函数：`my_ink, his_ink * -1`

#### **第2回合策略：**
- 考虑SP点数积累，为后续特殊攻击做准备
- 使用两步前瞻：评估当前行动后下一回合使用特殊攻击的可能性

#### **第3+回合策略（主要游戏阶段）：**
1. **特殊攻击优先**：当SP点数达到阈值时优先使用特殊攻击
2. **扩张阶段**（第5+回合）：尝试扩大领地，使用复杂的评分函数考虑：
   - 占领格子的数量（不同膨胀距离）
   - 冲突格子数量
   - 与对手的距离
   - 墨迹区域大小
   - SP点数变化
3. **巩固阶段**：巩固现有领地，评分考虑：
   - SP点数积累
   - 占领区域面积
   - 与特殊格的距离
   - 特殊牌使用惩罚

## 3. 评分系统核心组件

### Evaluation 类提供的关键评估函数：

1. **`occupied_grids()`**：计算有效占领格子数
   - 考虑己方和对手墨迹的膨胀效果
   - 使用连通组件分析确定孤立区域

2. **`conflict_grids()`**：计算冲突格子数
   - 评估扩张时可能发生的冲突

3. **`dilated_area()`**：计算膨胀后的区域面积
   - 用于评估领地的发展潜力

4. **`square_distance()`**：计算与对手墨迹的最小距离
   - 用于评估进攻的激进程度

5. **`ink_size()`**：计算墨迹区域的尺寸
   - 使用欧几里得距离衡量领地大小

## 4. 算法特点

### a) 多阶段策略
- **早期阶段**（1-2回合）：建立基础，积累SP
- **中期阶段**（3-4回合）：巩固阵地
- **扩张阶段**（5+回合）：积极扩张
- **特殊攻击阶段**：当SP足够时优先使用

### b) 启发式权重
算法使用精心调整的权重系数：
- 占领格子：权重 1.0
- 膨胀占领格子：权重 0.8
- 冲突格子：权重 -0.5
- SP点数：权重 6.0
- 对手SP变化：权重 -4.0

### c) 牌组管理
- 根据牌的大小和形状排序手牌
- 特殊牌（12格）有特殊处理逻辑
- 考虑牌的可用性和战略价值

## 5. 技术实现细节

### a) 图像处理
- 使用 OpenCV 进行图像处理
- 连通组件分析用于领地评估
- 形态学操作（膨胀）用于模拟墨迹扩散

### b) 状态估计
- `estimate_stage()`：估计行动后的游戏状态
- `estimate_status()`：估计完整游戏状态变化
- 支持单步和多步前瞻

### c) 性能优化
- 使用 NumPy 进行高效数组操作
- 选择性日志记录避免性能开销
- 缓存中间计算结果

## 6. 与其他AI算法的对比

### SimpleAI
- 完全随机选择行动
- 无策略性思考

### MCTS（蒙特卡洛树搜索）
- 基于随机模拟的搜索算法
- 需要大量计算资源
- 理论上更优但实际运行较慢

### Alpha算法的优势
- 实时性能好，适合实时游戏
- 基于领域知识的启发式规则
- 可解释性强，易于调试和调整

## 7. 实际应用效果
根据项目文档，Alpha算法能够：
- 击败 Level 3 NPC
- 实现自动化游戏，帮助玩家达到 Level 999
- 获取所有卡套（sleeves）

## 8. 改进空间

1. **机器学习集成**：可以结合强化学习优化权重参数
2. **对手建模**：增加对手行为预测
3. **长期策略**：考虑整局游戏的战略规划
4. **自适应调整**：根据对手水平动态调整策略

## 9. 文件结构

```
tableturf/ai/alpha/
├── __init__.py          # 模块初始化文件
├── alpha.py             # Alpha算法主实现
├── util.py              # 工具函数
└── Alpha算法分析.md     # 本文档
```

## 10. 关键代码片段

### 重抽决策逻辑
```python
def __score_redraw(hands: List[Card], stage: Optional[Stage] = None, 
                   my_deck: Optional[List[Card]] = None, 
                   his_deck: Optional[List[Card]] = None) -> float:
    hand_quantile = 2  # [0, 3]
    deck_quantile = 8  # [0, 14]
    cards = sorted([card for card in hands], key=lambda c: c.size)
    if my_deck is None:
        return 1 if cards[hand_quantile].size >= 10 else -1
    deck = sorted([card for card in hands + my_deck], key=lambda c: c.size)
    
    if cards[hand_quantile].size >= deck[deck_quantile].size:
        if np.all([util.is_special_card(card) for card in cards[hand_quantile:]]):
            return -1
        else:
            return 1
    else:
        return -1
```

### 扩张阶段评分函数
```python
def __score_expanding_step(current_score, status: Status, step: Step):
    occupied_grids_1, occupied_grids_2, occupied_grids_3, conflict_grids = current_score
    next_stage = util.estimate_stage(status.stage, step)
    estimated_occupied_grids_1 = Evaluation.occupied_grids(next_stage, my_dilate=1, his_dilate=1, connectivity=8) - occupied_grids_1
    estimated_occupied_grids_2 = Evaluation.occupied_grids(next_stage, my_dilate=1, his_dilate=3, connectivity=8) - occupied_grids_2
    estimated_occupied_grids_3 = Evaluation.occupied_grids(next_stage, my_dilate=2, his_dilate=1, connectivity=8) - occupied_grids_3
    estimated_conflict_grids = Evaluation.conflict_grids(next_stage, my_dilate=3, his_dilate=3) - conflict_grids
    size = Evaluation.ink_size(next_stage)
    my_sp = status.my_sp + util.estimate_my_sp_diff(status.stage, next_stage, step)
    his_sp_diff = util.estimate_his_sp_diff(status.stage, next_stage, step)

    pattern = step.card.get_pattern(step.rotate)
    distance = np.min([Evaluation.square_distance(status.stage, pos) for pos in pattern.offset + step.pos])
    if status.round > 11 and distance > 2:
        distance = distance * -100
    else:
        distance = distance * -1

    return estimated_occupied_grids_1, estimated_occupied_grids_2 * 0.8, estimated_occupied_grids_3 * 0.8, estimated_conflict_grids * -0.5, distance, size, my_sp * 6, his_sp_diff * -4
```

## 总结

Alpha算法是一个精心设计的基于规则的AI，它平衡了计算效率和游戏智能，是Autotableturf项目的核心组件之一。通过多阶段策略、启发式评分和实时状态估计，它能够在Tableturf Battle中做出合理的决策，实现自动化游戏。

---
*文档生成时间：2026-02-23*
*分析基于代码版本：Autotableturf项目最新版本*