# AutoTableTurf Rebuild

占地斗士自动化项目，包含离线规则模拟、自对弈、MCTS、牌组搜索，以及采集卡视觉识别和 EasyCon 控制的任务终端。

当前分支为 `v3`。实机流程仍在迭代，存在依赖本地截图和字体的识别配置；当前实现不能视为已经完成连续多局稳定性验收。

## 安装与检查

以下命令均在项目根目录执行。Python 版本范围为 3.11–3.13，依赖声明见 [pyproject.toml](pyproject.toml)。可使用已有 conda 环境：

```powershell
python -m pip install -e ".[dev,controller,vision]"
python -m tableturf_rebuild validate-data
python -m pytest -q
```

也可使用项目内虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,controller,vision]"
.\.venv\Scripts\python.exe -m tableturf_rebuild validate-data
```

文档中的 `python` 指已安装本项目的解释器；使用虚拟环境时可先激活，或替换为 `.\.venv\Scripts\python.exe`。双击启动器及长期运行脚本使用项目内 `.venv`。

## 启动工具

| 功能 | 命令 | 双击入口 |
| --- | --- | --- |
| 任务终端、牌组和实机自动对战 | `python -m tableturf_rebuild task-terminal` | `scripts/open_task_terminal.cmd` |
| 采集预览、录像和截图标注 | `python -m tableturf_rebuild capture-debug` | `scripts/open_capture_debug.cmd` |
| 视频回放与视觉分析 | `python -m tableturf_rebuild video-review` | `scripts/open_video_review.cmd` |
| ROI 标定与棋盘掩码编辑 | `python -m tableturf_rebuild calibrate-vision` | `scripts/open_calibration.cmd` |
| EasyCon / Dummy 手柄调试 | `python -m tableturf_rebuild controller-debug` | — |

任务终端需要采集卡、EasyCon 和已保存的 15 张预设牌组。操作顺序、暂停恢复、诊断和本机控制接口见 [任务终端](docs/TASK_TERMINAL.md)。

## 离线评估

```powershell
python -m tableturf_rebuild selfplay --games 10 --seed 42 --level 3
python -m tableturf_rebuild selfplay --games 10 --seed 42 --level 3 --policy-a mcts --policy-b greedy --simulations 32 --time-limit 1
```

对局记录默认写入 `artifacts/selfplay/`。长期续跑见 [RUN_LOOP](docs/RUN_LOOP.md)，牌组搜索见 [DECK_SEARCH](docs/DECK_SEARCH.md)。离线对手使用代理策略，结果不能直接当作真实 NPC 胜率。

## 文档导航

| 主题 | 文档 |
| --- | --- |
| 实机任务、调试、记牌与控制接口 | [TASK_TERMINAL](docs/TASK_TERMINAL.md) |
| 采集卡与视频审阅 | [CAPTURE_DEBUG](docs/CAPTURE_DEBUG.md) |
| EasyCon 协议与逐键验证 | [CONTROLLER_DEBUG](docs/CONTROLLER_DEBUG.md) |
| 视觉链路、配置及资源限制 | [VISION_PIPELINE](docs/VISION_PIPELINE.md) |
| ROI 与棋盘标定 | [ROI_CALIBRATION_SPEC](docs/ROI_CALIBRATION_SPEC.md) |
| 状态含义与人工修正 | [STATE_MACHINE](docs/STATE_MACHINE.md) |
| 模拟器规则与边界 | [RULES](docs/RULES.md) |
| MCTS 搜索方式与限制 | [MCTS](docs/MCTS.md) |
| 策略目标、评估和学习接入条件 | [AI_ROADMAP](docs/AI_ROADMAP.md) |
| 待验证的玩家策略经验 | [STRATEGY_NOTES](docs/STRATEGY_NOTES.md) |
| 牌组搜索、输出与后续扩展 | [DECK_SEARCH](docs/DECK_SEARCH.md) |
| 自对弈断点续跑 | [RUN_LOOP](docs/RUN_LOOP.md) |
| 数据审计与历史修复 | [DATA_AUDIT](docs/DATA_AUDIT.md) |
| 游戏数据来源与矩阵编码 | [DATA_SOURCES](src/tableturf_rebuild/data/DATA_SOURCES.md) |
| 自定义牌组格式 | [profiles/decks/README](profiles/decks/README.md) |

## 目录与本地资源

- `src/tableturf_rebuild/`：领域模型、搜索、视觉、控制器和应用层。
- `tests/`：离线单元测试与视觉回归测试。
- `profiles/vision/`：标定配置；`profiles/decks/`：用户牌组。
- `assets/`：状态模板、参考图和识别裁片；各素材目录的 README 保留来源说明。
- `tools/`、`scripts/`：诊断、数据同步与启动工具。
- `artifacts/`：本地录制、日志、实验输出和临时诊断，已由 Git 忽略。

部分 Profile 的参考图位于本地 `artifacts/`，字体识别默认读取本机 `E:\Splatoon\字体\`，卡面索引也引用仓库外的图片。新机器需要准备相应资源或重新标定，详见 [视觉链路](docs/VISION_PIPELINE.md)。

许可证文件见 [LICENSE](LICENSE)。

