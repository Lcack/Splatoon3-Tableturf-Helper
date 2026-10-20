# 发现与决策 — AutoTableTurf 项目分析

## 需求
- 用户通过 HDMI 将 Switch 连接到电脑
- 通过 OBS 或采集卡捕获 Switch 画面
- 图像识别来确认对手角色、已有卡组、对战棋盘
- AI 自动选择并放置卡牌进行对战
- 对战结束后根据练级需求循环继续

## 研究发现

### 1. 项目架构（良好）

```
video(capture) → detection(识别) → model(数据) → ai(决策) → action(宏) → controller(执行)
    ↑                                                                    ↓
    └──────────────── 状态反馈循环(l_manager.run()) ←─────────────────┘
```

| 层 | 路径 | 状态 |
|----|------|------|
| 采集 | `capture/` | 基础可用（OpenCV + FileLoader） |
| 控制器 | `controller/` | 支持 ESP32/NXBT/R3/Dummy 四种 |
| 数据模型 | `tableturf/model/` | 设计优秀（Grid/Card/Stage/Status/Step） |
| AI | `tableturf/ai/` | Alpha(启发式)+MCTS+SimpleAI |
| 检测 | `tableturf/manager/detection/` | 功能完整但坐标可能过期 |
| 动作 | `tableturf/manager/action/` | 完整但有 Bug |
| 管理 | `tableturf/manager/tableturf.py` | 核心编排器，较完整 |
| UI | `qt/` + `portal/` | 双界面（Qt桌面+Flask Web） |

### 2. 已知代码问题（详细列表）

#### 严重 Bug
| # | 文件 | 问题 | 影响 |
|---|------|------|------|
| 1 | `model/step.py:43` | `__hash__` 排除了 `rotate` 和 `pos` | 不同位置/旋转的 Step 哈希冲突 |
| 2 | `action/deck.py` | `// 8` 和 `% 8` 用于 15 张卡的网格 | 坐标映射可能错误 |
| 3 | `action/util.py` | `buttons_to_marco(down, up)` 参数未使用，硬编码 0.05s | 参数无效 |
| 4 | `controller/esp32.py:221` | 摇杆角度计算公式不标准 | 可能产生错误角度 |

#### 代码质量
| # | 问题 | 位置 |
|---|------|------|
| 5 | 全局 `marco` 拼写错误（应为 `macro`） | 所有 action 模块 |
| 6 | `logger.warn` 已弃用（应为 `warning`） | `action/card.py:188` |
| 7 | `redraw.py`/`replay.py`/`giveup.py` 完全重复 | action 模块 |
| 8 | `card.py` 中 `_fallback_hands()` 和 `EnhancedCardDetector.hands()` 代码重复 | detection |
| 9 | `video_widget.py` 中图像缩放代码重复 | qt/home |
| 10 | `EnhancedAlpha` 使用名称修饰访问父类私有方法 | `ai/alpha/enhanced_alpha.py` |
| 11 | Portal 使用全局变量（`capture`, `controller`, `closer`） | `portal/home/home.py` |
| 12 | 日志无文件输出、无轮转 | `logger/logger.py` |
| 13 | `file_loader.py` 文档说 .npy 但只加载 .jpg | `capture/file_loader.py` |

### 3. 参考项目分析

| 项目 | 用途 | 关键发现 |
|------|------|---------|
| **AutoSplatoon** (C++/Qt) | Splatoon 自动涂鸦 | ESP32串口协议实现(`serialcontroller.cpp`)、同步握手(Vanilla+Chocolate)、CRC8校验、按键位编码(`inputemulator.h`) |
| **UARTSwitchCon** | 通用 Switch 手柄模拟 | Protocol.md 定义完整协议，支持 AVR 和 ESP32 两套硬件 |

AutoTableTurf 已经将这两个项目的核心协议完整移植为 Python（`esp32_serial.py` + `esp32_input.py`）。

### 4. 游戏数据完整性

| 文件 | 内容 | 数量 |
|------|------|------|
| `MiniGameCardInfo_Matrix_ZH.json` | 卡牌数据（含中文名） | 266 张 |
| `MiniGameGameNpcData_ZH.json` | NPC 对手数据 | 30 个 |
| `MiniGamePresetDeck.json` | 预设卡组 | 31 个 |
| `MiniGameStage.json` | 战场地图 | 15 个 |
| `stage/*.png` | 战场预览图片 | 16 张 |

数据处理脚本（`merge_card_data.py`, `convert_to_matrix.py`, `merge_npc_data.py`）已将原始数据转换为 AI 可用格式。

### 5. 测试覆盖情况

| 模块 | 测试存在 | 通过 | 说明 |
|------|---------|------|------|
| capture | ✅ | ✅ FileLoader | VideoCapture 未验证 |
| controller | ✅ | 未知 | ESP32/NXBT 测试文件存在 |
| detection | ✅ | ❌ | 测试框架完整，但执行结果多为"占位符" |
| model | ✅ | 未知 | 有测试文件 |
| AI (alpha) | ✅ | 未知 | |
| AI (mcts) | ✅ | 未知 | |
| manager/action | ❌ | — | 无测试 |
| manager/closer | ❌ | — | 无测试 |
| UI (qt/portal) | ❌ | — | 无测试 |

## 技术决策

| 决策 | 理由 |
|------|------|
| 保留现有 model 层架构 | Grid/Card/Stage/Status/Step 设计合理，是整个系统的基石 |
| Alpha AI 作为主力策略 | 已有完整实现，能击败 Lv.3 NPC，实时性好 |
| MCTS 作为可选增强 | 需要更多计算资源，适合在性能充裕时使用 |
| 统一使用 Python logging | 当前混用 print 和 logger，需统一 |
| 采集方式支持 OBS 虚拟摄像头 | 只需 OpenCV 读取摄像头索引即可，`video_capture.py` 已实现 |
| 手柄控制优先 ESP32 方案 | 已有完整 Python 实现，比蓝牙方案更稳定 |
| 保留 Qt 作为主界面 | 功能更完整，Portal 作为轻量备选 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| 项目文件分散、混乱（历史遗留） | 按模块整理，移除冗余文件 |
| 检测坐标可能因游戏版本更新失效 | 需重新校准所有 ROI 坐标 |
| NXBT 蓝牙需 Linux 环境 | 推荐用户使用 ESP32 串口方案 |

## 资源
- [UARTSwitchCon Protocol.md](E:\Splatoon\tableturf\UARTSwitchCon-1.2\Protocol.md) — ESP32 通信协议规范
- [Alpha算法分析.md](E:\Splatoon\tableturf\AutoTableTurf\tableturf\ai\alpha\Alpha算法分析.md) — AI 策略文档
- [PROJECT_STRUCTURE.md](E:\Splatoon\tableturf\AutoTableTurf\PROJECT_STRUCTURE.md) — 现有结构文档
- [PROJECT_DEVELOPMENT_SUMMARY.md](E:\Splatoon\tableturf\AutoTableTurf\PROJECT_DEVELOPMENT_SUMMARY.md) — 开发历史

## 视觉/浏览器发现
- 游戏棋盘截图（`stage/*.png`）展示了 16 种战场布局：十字形、方形、钻石形、楼梯形等
- Qt 界面截图显示完整的控制面板、视频预览区、配置文件编辑器
- 项目 GitHub 有 [已知 Issue #9](https://github.com/fga401/AutoTableTurf/issues/9)：游戏 UI 更新导致检测坐标全部需更新

---
*每执行2次查看/浏览器/搜索操作后更新此文件*
*防止视觉信息丢失*
