# AutoTableTurf 项目结构

> Splatoon 3 Tableturf Battle 自动化AI助手 - 通过图像识别和蓝牙模拟实现自动对战

---

## 目录结构

```
AutoTableTurf/
├── 📁 .git/                          # Git版本控制目录
├── 📁 capture/                       # 视频捕获模块 - 采集卡视频源处理
│   ├── __init__.py                   # 模块初始化文件
│   ├── interface.py                  # 捕获接口定义
│   ├── video_capture.py              # OpenCV视频捕获实现
│   └── file_loader.py                # 从文件加载图像（测试用）
│
├── 📁 controller/                    # 控制器模块 - Switch蓝牙/串口控制
│   ├── __init__.py                   # 模块初始化
│   ├── interface.py                  # 控制器接口定义
│   ├── dummy.py                      # 虚拟控制器（测试用）
│   ├── nxbt.py                       # NXBT蓝牙控制器封装
│   ├── r3.py                         # R3控制器实现
│   ├── esp32.py                      # ESP32蓝牙控制器主类
│   ├── esp32_serial.py               # ESP32串口通信
│   ├── esp32_input.py                # ESP32输入处理
│   ├── esp32_flasher.py              # ESP32固件烧录工具
│   ├── test_esp32.py                 # ESP32测试脚本
│   └── 📁 nxbt_server/               # NXBT服务器（嵌入式）
│       ├── app.py                    # NXBT服务器入口
│       ├── requirement.txt           # NXBT依赖
│       ├── README.md                 # NXBT说明文档
│       ├── copy.sh                   # 复制脚本
│       └── 📁 nxbt/                  # NXBT核心库
│           ├── nxbt.py               # NXBT主模块
│           ├── bluez.py              # BlueZ蓝牙协议
│           ├── tui.py                # 终端UI
│           ├── logging.py            # 日志工具
│           └── 📁 controller/        # 控制器核心
│               ├── controller.py     # 控制器实现
│               ├── server.py         # 蓝牙服务器
│               ├── input.py          # 输入处理
│               ├── protocol.py       # 通信协议
│               ├── utils.py          # 工具函数
│               └── 📁 sdp/           # SDP配置文件
│                   └── switch-controller.xml
│
├── 📁 logger/                        # 日志模块
│   ├── __init__.py
│   └── logger.py                     # 日志配置和工具
│
├── 📁 portal/                        # Flask Web界面模块
│   ├── __init__.py
│   ├── 📁 home/                      # 主页路由
│   │   ├── __init__.py
│   │   ├── home.py                   # 主页逻辑
│   │   ├── capture.py                # 捕获相关路由
│   │   ├── keymap.py                 # 按键映射
│   │   ├── closer.py                 # 结束处理
│   │   └── 📁 templates/             # HTML模板
│   │       ├── home.html             # 主界面模板
│   │       └── home_esp32.html       # ESP32专用界面
│   ├── 📁 util/                      # 工具模块
│   │   ├── __init__.py
│   │   └── rwlock.py                 # 读写锁实现
│   ├── 📁 debug/                     # 调试页面
│   │   ├── __init__.py
│   │   ├── debug.py                  # 调试路由
│   │   └── 📁 templates/             # 调试模板
│   │       ├── debug.html
│   │       └── page.html
│   └── 📁 profiles/                  # 配置文件存储
│       └── default.json              # 默认配置
│
├── 📁 qt/                            # Qt桌面GUI模块
│   ├── __init__.py
│   ├── main.py                       # Qt应用入口
│   ├── run_qt.py                     # Qt启动脚本
│   ├── 📁 home/                      # 主界面组件
│   │   ├── __init__.py
│   │   ├── main_window.py            # 主窗口
│   │   ├── video_widget.py           # 视频显示组件
│   │   ├── control_widget.py         # 控制面板
│   │   ├── profile_widget.py         # 配置面板
│   │   └── controller_widget.py      # 控制器设置
│   ├── 📁 debug/                     # 调试窗口
│   │   ├── __init__.py
│   │   └── debug_window.py           # 调试窗口实现
│   └── 📁 util/                      # Qt工具
│       └── __init__.py
│
├── 📁 tableturf/                     # 核心游戏逻辑模块
│   ├── __init__.py                   # 导出核心类
│   │
│   ├── 📁 model/                     # 数据模型层 - 游戏对象定义
│   │   ├── __init__.py               # 导出模型类
│   │   ├── grid.py                   # 格子状态枚举(Empty/Ink/Special/Wall)
│   │   ├── card.py                   # 卡牌类 - 存储卡牌图案和SP消耗
│   │   ├── stage.py                  # 场地类 - 存储场地布局
│   │   ├── status.py                 # 游戏状态类 - 当前回合、手牌、SP等
│   │   ├── step.py                   # 行动步数类 - 卡牌、位置、旋转等
│   │   ├── data_loader.py            # JSON数据加载器（单例模式）
│   │   ├── card_factory.py           # 卡牌工厂 - 从JSON创建Card对象
│   │   ├── stage_factory.py          # 场地工厂 - 从JSON创建Stage对象
│   │   ├── MiniGameCardInfo_Matrix_ZH.json    # 266张卡牌数据
│   │   ├── MiniGameGameNpcData_ZH.json        # 30个NPC数据
│   │   ├── MiniGamePresetDeck.json            # 31个预设卡组
│   │   └── MiniGameStage.json                 # 15个场地数据
│   │
│   ├── 📁 ai/                        # AI决策层 - 各种AI算法实现
│   │   ├── __init__.py               # 导出AI类
│   │   ├── interface.py              # AI接口定义
│   │   ├── simple.py                 # 简单AI - 随机选择
│   │   │
│   │   ├── 📁 alpha/                 # Alpha启发式AI
│   │   │   ├── __init__.py           # 导出Alpha AI
│   │   │   ├── alpha.py              # 基础Alpha算法
│   │   │   ├── enhanced_alpha.py     # 增强版Alpha（集成JSON数据）
│   │   │   ├── util.py               # Alpha工具函数
│   │   │   └── Alpha算法分析.md      # Alpha算法文档
│   │   │
│   │   └── 📁 mcts/                  # MCTS蒙特卡洛树搜索AI
│   │       ├── __init__.py           # 导出MCTS AI
│   │       ├── mcts.py               # MCTS实现（带线性学习）
│   │       └── tableturf_rule.md     # Tableturf游戏规则文档
│   │
│   └── 📁 manager/                   # 管理层 - 游戏流程控制
│       ├── __init__.py               # 导出Manager
│       ├── tableturf.py              # 游戏主管理器
│       ├── data.py                   # 数据管理
│       │
│       ├── 📁 action/                # 动作执行
│       │   ├── __init__.py
│       │   ├── hands.py              # 手牌操作
│       │   ├── card.py               # 卡牌操作
│       │   ├── deck.py               # 卡组操作
│       │   ├── redraw.py             # 重抽逻辑
│       │   ├── giveup.py             # 放弃逻辑
│       │   ├── replay.py             # 回放逻辑
│       │   └── util.py               # 动作工具
│       │
│       ├── 📁 closer/                # 结束检测
│       │   ├── __init__.py
│       │   ├── interface.py          # 结束检测接口
│       │   ├── stats_closer.py       # 统计结束检测
│       │   └── union_closer.py       # 组合结束检测
│       │
│       └── 📁 detection/             # 图像检测模块
│           ├── __init__.py           # 导出检测函数
│           ├── card.py               # 卡牌检测（图像+JSON验证）
│           ├── stage.py              # 场地检测
│           ├── deck.py               # 卡组检测
│           ├── ui.py                 # UI状态检测
│           ├── util.py               # 检测工具
│           └── 📁 debugger/          # 调试工具
│               ├── __init__.py
│               ├── interface.py      # 调试接口
│               └── cv.py             # OpenCV调试器
│
├── 📁 tests/                         # 测试目录
│   ├── __init__.py
│   ├── 📁 test_capture/              # 捕获测试
│   │   ├── __init__.py
│   │   └── test_video_capture.py
│   ├── 📁 test_controller/           # 控制器测试
│   │   ├── __init__.py
│   │   ├── test_nxbt.py
│   │   └── test_interface.py
│   └── 📁 test_tableturf/            # 核心逻辑测试
│       ├── __init__.py
│       ├── 📁 test_model/            # 模型测试
│       │   ├── __init__.py
│       │   ├── test_card.py
│       │   ├── test_stage.py
│       │   └── test_status.py
│       ├── 📁 test_ai/               # AI测试
│       │   ├── __init__.py
│       │   ├── test_alpha.py
│       │   └── test_mcts.py
│       └── 📁 test_detection/        # 检测测试
│           ├── __init__.py
│           ├── test_card.py
│           ├── test_stage.py
│           ├── test_deck.py
│           └── test_ui.py
│
├── 📁 firmware/                      # 固件目录
│   └── README.md                     # ESP32固件说明
│
├── 📄 main.py                        # 项目主入口
├── 📄 run_portal.py                  # Web界面启动脚本
├── 📄 run_qt.py                      # Qt界面启动脚本
├── 📄 install_esptool.py             # esptool安装脚本
│
├── 📄 requirements.txt               # Python依赖列表
├── 📄 LICENSE                        # 开源许可证
├── 📄 README.md                      # 项目说明文档
├── 📄 .gitignore                     # Git忽略规则
│
├── 📄 PROJECT_DEVELOPMENT_SUMMARY.md # 开发总结文档
├── 📄 AUTOTABLETURF_QUICK_GUIDE.md   # 快速使用指南
└── 📄 PROJECT_STRUCTURE.md           # 本文件 - 项目结构说明
```

---

## 核心流程

```
视频捕获(capture) → 图像检测(detection) → AI决策(ai) → 控制器执行(controller)
        ↑                                              ↓
        └──────────── 状态反馈(model) ←───────────────┘
```

---

## 启动方式

| 方式 | 命令 |
|------|------|
| Qt桌面界面 | `python -m qt.main` 或 `python run_qt.py` |
| Web界面 | `python -m portal` 或 `python run_portal.py` |
| 主入口 | `python main.py` |

---

## 技术栈

- **图像处理**: OpenCV, NumPy
- **GUI**: PySide6 (Qt), Flask (Web)
- **AI算法**: MCTS, Alpha, 线性学习
- **控制器**: NXBT (蓝牙), ESP32 (串口)
- **数据**: JSON游戏数据文件

---

*生成时间: 2026-02-26*
