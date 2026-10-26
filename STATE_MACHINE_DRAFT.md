# 状态机草稿

来源：

```text
artifacts/state_capture/run-20261021-110258/
```

本次分析基于 13 条 JSON 标注、标注区域坐标，以及相邻截图的灰度差异。
当前会话没有直接把图片送入视觉模型的可用通道，因此图片内容判断以你的
文字标注和程序化像素统计为依据。

## 结论摘要

- 这 13 张图覆盖了从 NPC 选择到再战选择的主要正向流程。
- 截图序号不能直接当作状态：`002/003`、`006/007`、`007_1/007_2` 都属于同一类主画面，只是焦点或子模式不同。
- 必须采用两级识别：
  1. 主画面识别；
  2. 焦点、覆盖层和动态字段识别。
- `007_1` 和 `007_2` 应合并为同一个 `TARGET_PLACEMENT` 状态，用
  `placement_valid=False/True` 区分。
- `010` 是 `TURN_SELECT` 上的弃牌覆盖层。
- `011` 是 `TURN_SELECT` 上的特殊攻击模式。
- `012` 是 `TARGET_PLACEMENT` 的特殊攻击模式。
- `001 -> 002` 不是固定转移。非水母 NPC 中间存在对话状态，需要按两次
  A；水母类 NPC 可跳过该阶段。

## 像素差异证据

缩放到 64×36 灰度后的平均绝对差：

| 相邻截图 | 平均差 | 解释 |
|---|---:|---|
| 001 -> 002 | 71.36 | 主界面发生明显变化 |
| 002 -> 003 | 4.40 | 主画面相同，仅焦点变化 |
| 003 -> 004 | 72.51 | 进入卡组选择 |
| 004 -> 005 | 44.47 | 进入重抽阶段 |
| 005 -> 006 | 30.82 | 进入对战选牌 |
| 006 -> 007 | 5.62 | 同一对战画面，进入摆放子模式 |
| 007_1 -> 007_2 | 1.91 | 同一摆放状态，仅预览合法性不同 |
| 008 -> 009 | 5.53 | 同一对战画面，进入揭示/结算阶段 |
| 009 -> 010 | 26.34 | 出现弃牌覆盖层 |
| 010 -> 011 | 20.53 | 切换到特殊攻击覆盖层 |
| 011 -> 012 | 11.61 | 进入特殊攻击摆放 |
| 012 -> 013 | 46.52 | 进入对战结束/再战选择 |

所以不能使用单一全画面哈希识别所有状态。

## 规范状态

| 规范 ID | 现有截图 | 含义 | 主要动作 |
|---|---|---|---|
| `NPC_ROSTER` | 001 | NPC 列表与当前选中 NPC | 上下选择，A 确认 |
| `NPC_DIALOGUE` | 缺失 | 非水母 NPC 的对白 | A 跳过，预计两次 |
| `MATCH_SETUP` | 002, 003 | NPC 难度、场地和开始按钮 | 左右选难度，上下切焦点，A/B |
| `DECK_SELECT` | 004 | 卡组列表和卡组详情 | 方向键，A 确认 |
| `REDRAW` | 005 | 开局四张手牌，维持或重抽 | 左右选择，A 确认 |
| `TURN_SELECT` | 006 | 回合数、手牌、占地面积、SP、选牌 | 方向键，A |
| `TURN_SELECT_SKIP` | 010 | 跳过时选择弃牌 | 方向键，A，B 返回 |
| `TURN_SELECT_SPECIAL` | 011 | 特殊攻击卡牌选择模式 | 方向键，A，B 返回 |
| `TARGET_PLACEMENT` | 007_1, 007_2 | 普通卡牌摆放；合法性由预览颜色区分 | 方向键，X/Y，A，B |
| `SPECIAL_PLACEMENT` | 012 | 特殊攻击卡牌摆放 | 方向键，X/Y，A，B |
| `TURN_REVEAL` | 009 | 双方卡牌揭示与同回合结算 | 等待动画结束 |
| `MATCH_RESULT` | 缺失 | 胜负和最终占地面积 | 等待 |
| `REPLAY_CHOICE` | 013 | 再战或返回 | 左右选择，A 确认 |

## 关键转移

```text
NPC_ROSTER
  -- A -->
  if 对话界面:
      NPC_DIALOGUE -- A x 2 --> MATCH_SETUP
  else:
      MATCH_SETUP

MATCH_SETUP
  -- B --> NPC_ROSTER
  -- A (开始按钮已聚焦) --> DECK_SELECT

DECK_SELECT
  -- A --> REDRAW

REDRAW
  -- A --> TURN_SELECT

TURN_SELECT
  -- 选择卡牌并 A --> TARGET_PLACEMENT
  -- 选择跳过并 A --> TURN_SELECT_SKIP
  -- 开启特殊攻击并 A --> TURN_SELECT_SPECIAL

TARGET_PLACEMENT
  -- B --> TURN_SELECT
  -- A (合法预览) --> TURN_REVEAL

TURN_SELECT_SKIP
  -- 选择卡牌并 A --> TURN_REVEAL
  -- B --> TURN_SELECT

TURN_SELECT_SPECIAL
  -- 选择卡牌并 A --> SPECIAL_PLACEMENT
  -- B --> TURN_SELECT

SPECIAL_PLACEMENT
  -- B --> TURN_SELECT_SPECIAL
  -- A (合法预览) --> TURN_REVEAL

TURN_REVEAL
  -- 未到第 12 回合 --> TURN_SELECT
  -- 第 12 回合结束 --> MATCH_RESULT

MATCH_RESULT
  -- 等待奖励/升级弹窗结束 --> REPLAY_CHOICE

REPLAY_CHOICE
  -- 继续 --> DECK_SELECT
  -- 返回 --> MATCH_SETUP
```

## 分支注意事项

### NPC 对话

不要按 NPC 名称写死，也不要假设所有 NPC 固定两次 A。更稳妥的流程是：

1. 在 `NPC_ROSTER` 按 A。
2. 如果识别到对白界面，进入 `NPC_DIALOGUE`。
3. 最多按两次 A，每次后重新识别画面。
4. 一旦识别到 `MATCH_SETUP`，立即停止发送 A。

这样同时兼容水母类无对白 NPC、普通 NPC 和未来对话次数变化。

### MATCH_SETUP 焦点

`002` 与 `003` 主界面差异只有 4.40。必须单独识别：

- 难度等级区域是否高亮；
- 开始按钮是否变成墨汁形状；
- 左下黄色指针的位置。

不能把 002 和 003 当成两个完全不同的页面。

### 摆放合法性

`007_1` 与 `007_2` 主状态相同，差异只有 1.91。状态机不应靠整张图判断，
而应识别棋盘中央的预览颜色：

- 灰色预览：`placement_valid=False`，禁止按 A。
- 我方颜色预览：`placement_valid=True`，可以按 A。
- 特殊攻击模式还要确认预览与己方特殊点相邻。

### 胜负与奖励

`013` 已经标出再战选择，但当前没有单独标注：

- 胜利、失败、平局的完整结算界面；
- 双方最终占地面积；
- 等级提升或奖励弹窗；
- 再战选择的默认光标位置。

这些状态必须补充，否则流程可能在结算动画处误判。

## 还缺的截图或短视频

建议补充以下关键状态：

1. 非水母 NPC 的两段对话界面。
2. 水母 NPC 从按 A 到难度界面的无对白过程。
3. 胜利、失败、平局三种结算界面。
4. 升级、解锁卡包、获得奖励或没有奖励的弹窗。
5. `TURN_REVEAL` 完成后进入下一回合的稳定画面。
6. 卡组选择时禁用卡组与可选卡组的完整列表状态。
7. 弃牌覆盖层按 B 返回后的状态。
8. 特殊攻击按 B 返回普通选牌后的状态。
9. 第 12 回合最后摆放到结算之间的短视频。
10. 常见异常：断线、取消、无法操作、返回菜单。

## 识别层设计

第一版状态识别建议分三层：

1. 主画面分类：NPC 列表、准备、卡组、重抽、对战、结算。
2. 子模式分类：选牌、弃牌、特殊选牌、普通摆放、特殊摆放。
3. 动态字段：
   - 回合数；
   - 双方 SP；
   - 双方占地面积；
   - 剩余牌数；
   - 手牌位置与卡牌 ID；
   - 摆放预览合法性和格子位置。

状态机只接收带置信度的 `Observation`。低置信度或连续多帧不一致时暂停输入，
等待稳定画面，不直接进入下一状态。

## 第一版实现状态

已从两轮标注生成 54 个状态模板，覆盖以下规范状态：

```text
NPC_ROSTER
NPC_DIALOGUE
MATCH_SETUP
DECK_SELECT
BATTLE_INTRO
REDRAW
TURN_SELECT
SKIP_DISCARD
SPECIAL_SELECT
TARGET_PLACEMENT
SPECIAL_PLACEMENT
TURN_REVEAL
TURN_RESOLVE
FINISHING
MATCH_RESULT
MATCH_REWARD
REPLAY_CHOICE
```

已在完整录像上顺序运行状态机：

```text
标注帧总数：41
识别正确：37
顺序准确率：90.2%
```

主要剩余误判：

- 菜单切换动画尚未完全稳定时，卡组界面可能短暂识别为对战介绍。
- 难度界面刚出现时可能仍沿用上一帧的 NPC 列表状态。
- 弃牌覆盖层和特殊攻击覆盖层相邻切换时可能互相混淆。

这些误判都在相邻帧内自行恢复，但正式输入前必须增加“稳定画面”条件，
不能在切换帧直接执行按键。

## 状态名称对照

| 英文状态 | 中文含义 |
|---|---|
| `UNKNOWN` | 未知画面 |
| `NPC_ROSTER` | NPC 列表 |
| `NPC_DIALOGUE` | NPC 对白 |
| `MATCH_SETUP` | NPC 难度与开始按钮 |
| `DECK_SELECT` | 卡组选择 |
| `BATTLE_INTRO` | 对战开场动画 |
| `REDRAW` | 重抽手牌 |
| `TURN_SELECT` | 回合选牌 |
| `SKIP_DISCARD` | 跳过并选择弃牌 |
| `SPECIAL_SELECT` | 特殊攻击选牌 |
| `TARGET_PLACEMENT` | 普通卡牌摆放 |
| `SPECIAL_PLACEMENT` | 特殊攻击卡牌摆放 |
| `TURN_REVEAL` | 双方亮牌和大小比较 |
| `TURN_RESOLVE` | 落墨、重叠与特殊点结算 |
| `FINISHING` | 最终结算动画 |
| `MATCH_RESULT` | 胜负结算 |
| `MATCH_REWARD` | 经验与奖励 |
| `REPLAY_CHOICE` | 是否继续对战 |

## 对战阶段修正规则

修正时标记当前画面，不标记准备执行的下一个状态：

- 手牌和卡牌详情可操作，棋盘上还没有预览：`TURN_SELECT`。
- 已选卡并出现棋盘预览，无论预览合法还是灰色：`TARGET_PLACEMENT`。
- 跳过操作打开了弃牌选择：`SKIP_DISCARD`。
- 已进入特殊攻击，正在选择可用卡牌：`SPECIAL_SELECT`。
- 特殊攻击卡已经出现在棋盘预览中：`SPECIAL_PLACEMENT`。
- 双方卡牌翻开并显示大小/放置顺序：`TURN_REVEAL`。
- 墨迹扩散、覆盖、特殊点激活：`TURN_RESOLVE`。
- 最后一回合结束并出现 `Finish` 提示：`FINISHING`。
- 已显示胜利或失败结果：`MATCH_RESULT`。
- 显示经验、奖励或升级：`MATCH_REWARD`。
- 显示继续或取消再战：`REPLAY_CHOICE`。

## 视频中修正误判

不需要修改帧号。打开视频审阅窗口后：

1. 暂停在误判帧。
2. 在右侧“修正状态”下拉框选择正确状态。
3. 点击“以当前帧修正并保存模板”。
4. 当前帧会加入该状态的困难样本。
5. 识别器立即重新加载，状态机当前状态同步修正。

人工模板保存在：

```text
assets/state_templates/manual/STATE_NAME/
```

索引保存在 `assets/state_templates/manual_manifest.json`。重复生成基础模板
不会覆盖这些人工修正样本。

