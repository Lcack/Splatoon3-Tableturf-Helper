# 状态机与人工修正

状态定义、显示名与允许转移以 [state_machine.py](../src/tableturf_rebuild/application/state_machine.py) 中的 `GameState`、`TRANSITIONS` 为准；自动按键与后置确认由任务运行器和对战执行器处理。

## 状态含义

标注当前画面已经发生的状态，不能把准备执行的下一步当作当前状态。

| 状态 | 含义 |
| --- | --- |
| `UNKNOWN` | 未知画面 |
| `NPC_ROSTER` | NPC 列表 |
| `NPC_DIALOGUE` | NPC 对白 |
| `MATCH_SETUP` | 难度、场地与开始按钮 |
| `DECK_SELECT` | 卡组选择 |
| `BATTLE_INTRO` | 对战开场动画 |
| `REDRAW` | 重抽或维持手牌 |
| `TURN_SELECT` | 回合选牌，尚未出现摆放预览 |
| `SKIP_DISCARD` | 已打开跳过，正在选择弃牌 |
| `SPECIAL_SELECT` | 已开启特殊攻击，正在选牌 |
| `TARGET_PLACEMENT` | 普通摆放，包含合法及灰色无效预览 |
| `SPECIAL_PLACEMENT` | 特殊攻击摆放 |
| `TURN_REVEAL` | 双方亮牌 |
| `TURN_RESOLVE` | 落墨、覆盖与特殊点结算 |
| `FINISHING` | 最终结算动画 |
| `MATCH_RESULT` | 胜负结果 |
| `MATCH_REWARD` | 经验、奖励与通知 |
| `REPLAY_CHOICE` | 是否继续对战 |

## 主流程

```text
NPC_ROSTER -> NPC_DIALOGUE（可跳过）-> MATCH_SETUP -> DECK_SELECT
  -> BATTLE_INTRO -> REDRAW -> TURN_SELECT

TURN_SELECT -> TARGET_PLACEMENT -> TURN_REVEAL
TURN_SELECT -> SKIP_DISCARD -> TURN_REVEAL
TURN_SELECT -> SPECIAL_SELECT -> SPECIAL_PLACEMENT -> TURN_REVEAL

TURN_REVEAL -> TURN_RESOLVE -> TURN_SELECT（下一回合）
TURN_RESOLVE -> FINISHING -> MATCH_RESULT
MATCH_RESULT -> MATCH_REWARD（可跳过）-> REPLAY_CHOICE
REPLAY_CHOICE -> DECK_SELECT（继续）或 MATCH_SETUP（退出）
```

这是主流程示意，取消、漏帧和恢复分支以代码为准。页面状态与焦点、手牌、特殊攻击开关、预览合法性分别读取；同一页面不能仅凭全图相似度决定发送按键。

## 自动执行的确认边界

- 画面识别和控制器发送成功都不等于动作完成，需要后续画面证据。
- 灰色预览仍属于摆放状态，不能据此确认放置；目标还需通过规则核验。
- 亮牌或结算样式的画面不能单独消耗回合数，执行器需要本局已提交动作的凭据。
- 比赛结果、经验奖励和继续对战有独立读数器；页面跳转还受稳定帧、间隔和重试次数约束。
- 计胜与返回菜单后的胜场核对由任务运行器处理，不能从某一张结果模板推断任务已经完成。

最新阈值、恢复行为和核心重载的状态保留规则见 [TASK_TERMINAL](TASK_TERMINAL.md)。

## 视频中修正误判

```powershell
python -m tableturf_rebuild video-review --video "录像路径.mp4"
```

1. 暂停在误判帧。
2. 在右侧“修正状态”选择正确状态。
3. 点击“以当前帧修正并保存模板”。
4. 检查识别结果与相邻帧，避免把切换动画标成稳定画面。

人工样本位于 `assets/state_templates/manual/STATE_NAME/`，索引为 `assets/state_templates/manual_manifest.json`。基础模板生成与人工样本的读写逻辑见 [state_machine.py](../src/tableturf_rebuild/application/state_machine.py)。

## 维护要求

新增状态时同步维护枚举、中文显示名、转移、视觉证据、执行器处理和回归测试。新样本应覆盖焦点变化、动画和相似页面；单段录像上的历史准确率不能作为当前版本的验收结果。
