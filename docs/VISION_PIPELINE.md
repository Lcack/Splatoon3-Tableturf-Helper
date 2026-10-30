# 视觉链路与资源配置

本文记录当前源码结构与资源入口。具体自动执行和失败恢复见 [TASK_TERMINAL](TASK_TERMINAL.md)，标定步骤见 [ROI_CALIBRATION_SPEC](ROI_CALIBRATION_SPEC.md)。所有命令在项目根目录执行。

## 当前链路

`CaptureWorker` 采集画面，`BattleVisionRuntime` 组合页面状态和各读数器，任务运行器及对战执行器使用识别结果决定后续动作。

| 部分 | 主要实现 | 用途 |
| --- | --- | --- |
| 页面分类 | `application/state_machine.py`、`vision/profile_state_verifier.py` | 全图候选、ROI 证据和时序稳定 |
| 配置语义 | `vision/profile_semantics.py`、`vision/selection_profiles.py` | 解析区域角色、选择手牌焦点几何 |
| 棋盘 | `vision/board_reader.py`、`vision/legacy_grid.py` | 网格定位、格类型和特殊点观察 |
| 卡牌 | `vision/hand_reader.py`、`vision/deck_reader.py`、`vision/reveal_reader.py` | 手牌、牌组及亮牌识别 |
| 摆放预览 | `vision/placement_reader.py`、`vision/preview_features.py`、`vision/legacy_preview.py` | 预览纹理、位置、旋转和有效性 |
| 模式与 SP | `vision/battle_mode_reader.py`、`vision/skip_mode_reader.py`、`vision/sp_reader.py` | 特殊攻击、弃牌与可用 SP |
| NPC 菜单 | `vision/npc_menu_reader.py`、`vision/font_npc_name.py`、`vision/font_digits.py` | NPC、难度、胜场和场地 |
| 结算 | `vision/match_result_reader.py`、`vision/reward_page_reader.py`、`vision/replay_choice_reader.py` | 胜负、经验通知与继续对战 |

这些模块的存在表示对应链路已实现，不表示所有页面和硬件配置均经过实机验收。低置信度、动画和候选冲突的处理条件以读数器与执行器代码为准。

## Profile 与棋盘

[profiles/vision/index.json](../profiles/vision/index.json) 包含：

- `state_profiles`：页面状态对应的 ROI 配置。
- `selection_geometry`：普通手牌与四个焦点变体。
- `board_profiles`：场地 ID 到棋盘标定文件的映射。
- `result_profiles`、`score_profiles`：比赛结果与比分配置。

当前索引包含 15 个场地的棋盘 Profile。Profile 可以带 `stage_id`、行列数与 `cell_overrides`；棋盘几何和掩码需对照当前采集画面验证，不能把“已有配置”当作实机识别正确的证明。

角点、手牌网格、成本灯和颜色样本通过标定器维护，避免直接修改散落坐标：

```powershell
python -m tableturf_rebuild calibrate-vision
```

格子类型与原始数据编码见 [DATA_SOURCES](../src/tableturf_rebuild/data/DATA_SOURCES.md)。UI 参考图的语义映射另见 [semantic_manifest.json](../assets/reference/stage/semantic_manifest.json)，不能把图片序号直接当作矩阵编码。

## 卡牌与 SP 的确认

手牌、牌组识别使用图案和成本信息，并结合当前牌组约束。独立的 `CardArtMatcher` 可用于卡面资源匹配，不能据此宣称所有实机路径都已采用卡面匹配。

焦点、灰牌状态和卡牌身份需要分别判断。SP HUD 可花费余额与棋盘上已激活特殊点数量分别读取；两者不能互相替代。预览的直接纹理、历史观察和底色提示也需区分，避免把背景当成牌形。

## 命令行与视频分析

无窗口实时分析：

```powershell
python -m tableturf_rebuild vision-live --backend DSHOW --device-index 0 --width 1920 --height 1080 --fps 60 --stage Rectangle --max-seconds 30 --output artifacts/live/vision.jsonl
```

单帧棋盘检查：

```powershell
python -m tableturf_rebuild inspect-board --image "截图路径.png" --profile "profiles/vision/对战场地-正直大道.json" --stage Rectangle
```

视频审阅与状态修正：

```powershell
python -m tableturf_rebuild video-review --video "录像路径.mp4"
```

正在运行的任务应通过 `tools/read_live_task.py` 和 `tools/control_live_task.py` 读取已有采集与诊断，详细用法见 [任务终端](TASK_TERMINAL.md)。

## 本地资源限制

Git 仓库保留 Profile、状态模板和识别裁片，但 `artifacts/` 被忽略，部分 Profile 的源截图仍指向该目录。新机器需恢复相应参考图或重新标定；仅安装 Python 包不能补齐这些资源。

默认字体路径在 [font_digits.py](../src/tableturf_rebuild/vision/font_digits.py) 与 [font_npc_name.py](../src/tableturf_rebuild/vision/font_npc_name.py) 中，当前指向本机 `E:\Splatoon\字体\`。字体不在本仓库中，缺失时需自行配置识别器的字体参数或调整本地路径。

卡面与卡背索引见 `assets/card_assets/`，构建工具为 [build_card_asset_index.py](../tools/build_card_asset_index.py)，原图片默认在仓库外的 `E:\Splatoon\splat3\images\minigame\`。素材来源说明保留在各 `assets/` 子目录的 README 中。

## 后续验证

以下是待完成的工程验证，不是当前能力声明：

- 逐场地核验棋盘网格、掩码、预览及复杂底色。
- 为识别冲突和人工接管保留可重复回放的样本。
- 增加不同等级、奖励、通知和动画相位的实机覆盖。
- 移除或配置化本地绝对路径，验证新机器上的资源准备流程。
- 分别报告视觉错误、执行错误和策略效果，不以少量参考帧结果推断连续运行稳定性。
