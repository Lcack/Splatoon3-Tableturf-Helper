# 视觉识别升级方案

## 可沿用的旧项目思路

`AutoTableTurf-master` 的识别细度来自专用局部算法，不来自一个全局分类器：

- UI 光标：固定 ROI + HSV 像素比例。
- 手牌/卡组：逐张卡的 8×8 格子采样 + 6 个 SP 成本位置。
- 棋盘：从浅色空格连通域恢复网格间距和全部格点。
- 棋盘格：按 HSV 分类空地、双方普通墨、特殊点、燃烧特殊点和墙。
- 摆牌预览：逐格检查 Canny 斜线纹理，再结合格子底色分类。
- SP：固定指示灯 ROI 计数。
- 多帧检测：连续识别结果一致后才执行动作。

这些算法可以对照当前终版 UI 重写，不能直接复制旧坐标。

## 固定版本假设

当前可以先建立单一参考配置：

- 分辨率：1920×1080。
- 我方颜色：黄绿色。
- 敌方颜色：蓝紫色。
- 游戏 UI：Splatoon 3 终版界面。
- 采集卡：Hagibis。

即使版本固定，ROI 仍应放入版本化 profile，避免因采集卡、驱动、缩放或 UI
细节变化而修改 Python 源码。

## ROI 精度调整

已提供高精度标定工具：

```powershell
.\.venv\Scripts\tableturf.exe calibrate-vision
```

工具支持：

- 打开任意 1920×1080 截图。
- 放大和缩小。
- 实际像素坐标显示。
- 两点矩形。
- 四点透视四边形。
- 自动生成行×列子网格。
- 区域列表、覆盖删除和覆盖重画。
- 导出版本化 JSON profile。
- 重新加载已有 profile。
- 导出带 ROI 和子网格的 `-overlay.png` 检查图。

当前状态标注是用于解释画面的粗框，不能直接用于 8×8 格子采样。精确 ROI
应采用一次性标定工具：

1. 加载一张稳定 1920×1080 截图。
2. 进行放大显示和像素坐标读数。
3. 用四点多边形标出卡牌、棋盘、胜场数字等区域。
4. 导出相对于 1920×1080 的像素坐标和归一化坐标。
5. 在原图上叠加采样格与 mask，检查每个采样格是否落在正确位置。
6. 用多帧截图验证，而不是只检查一张图。

不同元素应使用不同几何：

- 卡牌和倾斜棋盘使用四点透视多边形。
- 数字和 SP 灯使用矩形。
- 棋盘格使用由四个角点推导的规则网格。
- 光标候选使用小矩形。

建议 profile：

```text
profiles/vision/splat3-final-1920x1080.json
```

内容包括：

```text
board_corners
hand_card_polygons
focus_card_polygon
deck_card_polygons
win_count_rois
sp_indicator_rois
ui_cursor_rois
color_samples
```

每个区域保存边框内缩量，避免卡牌边框、阴影和抗锯齿影响。

## 卡牌识别

解包资源：

```text
E:\Splatoon\splat3\images\minigame\card
E:\Splatoon\splat3\images\minigame\sleeve
```

核对结果：

- 卡面 PNG 共 267 个。
- 其中 266 个文件名与 266 张卡牌的 `Name` 一一对应。
- 额外文件为 `Mng_CardDummy_00.png`。
- 卡背 PNG 共 32 个，31 个正式资源加 1 个 Dummy。
- NPC 数据中的 `CardSleeve` 可以直接映射到卡背名称。

建议采用三级卡牌识别：

### 第一层：格子图案与成本

- 当前手牌/放大卡使用透视校正。
- 采样 8×8 格。
- 分类普通墨、特殊墨、空。
- 读取 6 个 SP 成本指示位置。
- 生成 `(pattern, sp_cost, area)`。

### 第二层：牌堆候选过滤

- 我方只搜索已知牌组和未出牌候选。
- 敌方使用 NPC 对应牌组作为候选。
- 去掉已经确认打出或弃掉的卡牌。

这通常能把 266 张卡牌缩小到十几张。

### 第三层：卡面图像身份确认

- 将游戏内卡面透视校正到解包卡图的统一尺寸。
- 使用卡图主体区域做 ORB/AKAZE/SIFT 特征匹配。
- 同时比较颜色直方图、边缘和感知哈希。
- 支持 0°、90°、180°、270° 旋转。
- 只有连续多帧高置信度一致时才确认卡牌 ID。

8×8 图案和 SP 成本不能去掉。卡图匹配负责身份，图案负责布局和规则校验；
两者不一致时返回低置信度。

## 卡背识别

卡背不属于卡牌身份，只用于判断：

- 对方牌是否仍为背面。
- 当前是否处于亮牌阶段。
- 牌面卡是否完整露出。

NPC 的 `CardSleeve` 已加入领域数据，后续可从 NPC 数据确定对手卡背。

## 摆牌预览与状态机

状态机不能只靠整图判断摆牌阶段。应输出独立证据：

```text
hand_selection_active
placement_preview_present
placement_preview_valid
placement_preview_cells
special_attack_mode
reveal_cards_visible
```

MCTS 只有在以下条件满足时才进入落点规划：

```text
state == TARGET_PLACEMENT
and placement_preview_present
```

## 胜场与记牌

- 三档胜场使用数字 OCR 或固定数字模板读取。
- 已出卡牌使用卡面识别。
- 我方剩余牌由牌组减手牌、已出牌和弃牌得到。
- 敌方剩余牌由已知 NPC 牌组减已确认出牌和弃牌得到。
- 未经高置信度确认的卡不得从剩余牌中删除。

## 2026-10-22 已实现部分

### Profile 索引与语义

阶段与 Profile 的对应关系保存在：

```text
profiles/vision/index.json
```

索引包含：

- `state_profiles`：每个状态对应的精细参考 Profile。
- `selection_geometry`：未选牌和四个聚焦位置的选牌几何。
- `board_profiles`：场地 ID 对应的棋盘透视 Profile。

`ProfileSemantics` 会把自由命名的区域解析为棋盘、手牌外框、8×8 网格、
点数、SP 成本、双方 SP、结算卡牌、胜场和 NPC 信息等语义角色。后续算法
不应继续依赖完整中文区域名。

### 状态机

状态机采用两级判断：

1. 全画面模板给出前四候选。
2. 只有第一、第二候选分差小于 `0.12` 时，才读取最多四个候选状态的精细
   ROI。

稳定画面不会每帧执行全部 ROI。含 42 张已加载截图的内存基准约为
`177 fps`，因此视频顺序测试的耗时主要来自 1080p H.264 解码，而不是分类器。

新增 `StateEvidence.placement_preview_present`。当棋盘检测明确存在摆牌预览
时，加强 `TARGET_PLACEMENT`/`SPECIAL_PLACEMENT` 并压低同一帧的
`TURN_REVEAL`/`TURN_RESOLVE` 候选。预览不存在时不反向推断状态。

### 选牌几何

`SelectionProfileResolver` 可在以下变体中选择最接近的 Profile：

```text
idle
focus_1_1
focus_1_2
focus_2_1
focus_2_2
```

`focus_2_1` 的临时源图已切换到 `026_video_state.png`。该画面实际属于
`SPECIAL_SELECT`，因此该 Profile 同时用于特殊攻击选牌几何和状态 ROI 证据。

四张卡的 `HAND_CARD` 区域还会计算亮度：聚焦卡相对其他三张的平均亮度差
作为独立特征。当前截图中 `focus_1_1`、`focus_1_2`、`focus_2_1`、
`focus_2_2` 的亮度差分别约为 `+39`、`+52`、`+35`、`+36`。用户会重新截取
普通选牌画面中的 `focus_2_1`，当前 `026` 配置作为临时来源。

### 棋盘

`BoardGridReader` 一次透视校正整张棋盘，再采样所有格子。每个格子输出：

- 原始关卡格类型；
- 推断格类型；
- BGR/HSV；
- 我方和对方颜色比例；
- 与空棋盘参考的色差；
- 边缘密度；
- 摆牌预览分数和是否存在预览。
- 初始棋盘是否已经锁定。

命令行检查：

```powershell
.\.venv\Scripts\tableturf.exe inspect-board `
  --image "artifacts\state_capture\run-20261021-221749\session-20261021-222704_annotations\017_video_state.png" `
  --profile "profiles\vision\对战场地-正直大道.json" `
  --stage Rectangle
```

连续帧验证中，第 17 帧双方各 1 个初始特殊点；第 19、20、21、22 帧的
已占格数量持续上升，符合落墨过程。第 1971 帧的高边缘格为 16，而第 2008
帧只剩 2，预览证据能够区分摆放画面和亮牌后的稳定画面。

### 当前限制

- 棋盘透视 Profile 目前只标定了 `Rectangle`，其余 14 个场地需要各自
  标定棋盘四边形和行列数。
- 摆放预览已能检测“存在”，尚未可靠分类“灰色无效”和“黄色有效”。
- 手牌、卡组、结算卡的 ID 尚未接入 SIFT 匹配。
- 胜场数字、面积数字、SP 成本和双方 SP 灯尚未加入具体读数器。

## 实时与视频分析入口

### 无窗口实时采集

采集卡视觉链路使用 `CaptureWorker` 和 `BattleVisionRuntime`，不会创建 Tk
窗口，也不会调用 OpenCV 窗口函数：

```powershell
.\.venv\Scripts\tableturf.exe vision-live `
  --backend DSHOW `
  --device-index 0 `
  --width 1920 `
  --height 1080 `
  --fps 60 `
  --stage Rectangle `
  --max-seconds 30 `
  --output artifacts\live\vision.jsonl
```

`--max-seconds` 和 `--max-frames` 都是可选的限时/限量保护。正式运行时可用
`Ctrl+C` 停止。输出包含状态、置信度、双方占格、摆牌预览和选牌变体。

### 视频视觉分析窗口

```powershell
.\.venv\Scripts\tableturf.exe video-review --video "<录像路径>"
```

窗口在原有播放、暂停、逐帧和修正模板功能上增加：

- 状态候选和精细 ROI 证据；
- 棋盘格推断覆盖；
- 我方/对方已占格数量；
- 摆牌预览分数；
- 选牌变体 Profile 和四个卡牌几何；
- 单帧前进保留状态历史，后退和拖动时间轴重置状态。

当前“分析覆盖”默认开启。棋盘填充颜色：

```text
黄绿色：我方普通墨
黄色：我方特殊点
蓝紫色：对方普通墨
青色：对方特殊点
灰色：墙
高边缘黄框：疑似摆牌预览
```

棋盘读取只在稳定进入 `TURN_SELECT`、弃牌/特殊选牌、摆放、亮牌或落墨阶段
时启动。NPC、难度、卡组、重抽和对白阶段不会因为全图候选里出现战斗状态而
触发棋盘 ROI。

颜色语义来自：

```text
assets/reference/stage/semantic_manifest.json
```

`AreaPiece_00` 到 `AreaPiece_04` 分别表示我方普通、敌方普通、我方特殊、
敌方特殊和无效格，`Mng_Point_02` 表示空格。敌方特殊 Hue 约 `92`，此前
落在普通敌方 Hue 范围之外，当前版本已经独立识别。

棋盘观察同时输出：

```text
self_special_count
opponent_special_count
initial_board_ready
initial_board_locked
```

特殊点数量来自整张棋盘矩阵，不只是初始两个静态点；卡片铺出的动态特殊格
也会计入。

总特殊点和激活特殊点分两层判断：

- 总特殊点：关卡静态特殊点、普通特殊 tile 颜色、特殊 tile 模板差值。
- 激活特殊点：火焰中心持续亮核、中心饱和度/亮度对比，并保留短暂时序状态。

实测参考区间：

```text
video: session-20261021-222704.mp4
time:  75.900s - 79.667s
frames: 2277 - 2390
```

该区间真实状态为我方总特殊5/激活1、敌方总特殊5/激活3。当前逐帧结果为
`114/114` 帧一致。火焰外轮廓会周期性熄灭，但中心亮点持续存在；敌方激活格
中心饱和度约 `10–22`，普通特殊格约 `205`。我方激活格中心相对整格亮度对比
约 `+9–13`，普通特殊格约 `0–2`。

已激活特殊点使用额外的高亮特征识别：

```text
我方激活特殊攻击点.png
敌方激活特殊攻击点.png
```

激活格通常比普通特殊格饱和度更低，并有明显高亮边缘。棋盘读取输出：

```text
self_active_special_count
opponent_active_special_count
```

视频窗口使用以下紧凑格式：

```text
我方52 特殊1：敌方45 特殊3
```

### 胜负判断

已新增：

```text
profiles/vision/比赛结果.json
```

该 Profile 复用 `011_video_state.png` 已标注的上下两个 WIN/LOSE 区域。
`MatchResultReader` 以这两块作为 WIN、LOSE 原型，在 `MATCH_RESULT` 状态
判断上方对手和下方我方结果：

```text
对手 WIN + 我方 LOSE -> 我方失败
对手 LOSE + 我方 WIN -> 我方胜利
无法可靠分配 -> UNKNOWN
```

视频窗口会用红框标出 WIN、绿框标出 LOSE，并显示比赛结果。

### 当前分数 ROI

建议额外增加一套 `对战HUD.json`，只包含：

```text
我方分数
敌方分数
```

区域应只框数字，不带图标或背景装饰。HUD 分数用于独立校验棋盘识别：

```text
棋盘识别面积与 HUD 分数差得很多 -> 漏格、误判或特殊点错误
```

当前 `ProfileSemantics` 已支持 `SELF_SCORE` 和 `OPPONENT_SCORE` 语义，
但具体数字读取器需要该 ROI 标定后才能校准。没有 HUD ROI 时，仍使用棋盘
格数作为估计分数。

当前 `比分情况.json` 已经接入 `ScoreTemplateReader`。已确认参考帧为
`我方52`、`敌方48`，可读取数字 `2/4/5/8`。没有对应模板的数字会返回 `?`；
此时窗口回退到棋盘格数，并以 `~` 表示估计值。

### 字体与文本识别

本机字体：

```text
E:\Splatoon\字体\华康新综艺体W7-P.ttf
E:\Splatoon\字体\华康新综艺体W9-P.ttf
E:\Splatoon\字体\SplatoonFontFix.otf
```

中文 NPC 名称不需要通用 OCR。NPC 名称只有 30 个已知候选，适合渲染候选文本
后做闭集匹配。英文和数字优先使用 `SplatoonFontFix.otf`。

当前 `FontNpcNameRecognizer` 使用 `华康新综艺体W7-P.ttf`，把游戏中的窄高汉字
逐字拉伸到统一方形，再与 30 个已知 NPC 名逐字比较。已确认：

```text
001_video_state.png -> 武器子
003_video_state.png -> 新手水母
```

难度选择界面把 `难度选择` ROI 横向分成三段，用亮度和边缘差异找高亮难度。
`003_video_state.png` 输出难度2。

胜场数字使用 `SPLATOON2.OTF` 生成完整的 `0–9` 字形模板，与对战比分使用的
截图模板分开。`REDRAW` 不属于卡牌聚焦状态，不运行选牌几何解析。

地图名也使用同一 `FontClosedSetRecognizer`，候选为 15 个已知中文场地名。
`003_video_state.png` 的 `对战场地名` 输出“正直大道”，并映射为
`Rectangle`。

`NpcMenuReader` 输出：

```text
npc_id
npc_name
level_wins
difficulty
stage_id
stage_name
binding_consistent
```

如果难度已知，地图必须匹配该 NPC 对应难度的 `map_ids`。绑定冲突时地图 ID
会被拒绝，也不会构造 `OpponentContext`。

当 NPC、难度和地图都获得高置信度时，`BattleVisionRuntime` 构造
`OpponentContext`，包含 NPC AI 类型、NPC 等级牌组和场地 ID。通过
`BattleVisionRuntime(..., mcts_policy=policy)` 或 `bind_mcts(policy)` 可将
上下文传给 `MCTSDecisionPolicy.set_opponent_context()`。该信息在进入牌组
选择后继续保留。当前行为模型仍默认 Greedy，因此 NPC AI 类型目前只作为
可用的对手上下文，不会假装已经具备对应的行为差异。

### 牌组点数

`DeckGridReader` 从 `卡组选择.json` 提取15张卡的8×8图案和成本灯，在8×8
画布内滑动卡牌裁剪矩阵并尝试旋转/翻转，然后用31个预设牌组做整组消歧。
视频和采集窗口显示：

```text
最佳牌组
15张卡的面积点数
每张卡的SP成本
面积总点数
SP成本总和
```

当前参考卡组识别为 `MiniGame_Starter`，总面积为 `131`，SP总成本为 `51`。
`MiniGame_Starter` 和 `MiniGame_Kurage0` 卡牌集合相同，牌组 ID 可能并列。

棋盘 Profile 可保存 `stage_id` 和 `cell_overrides`。在标定器中开启格子编辑后，
可显式选择 `EMPTY/SELF_INK/SELF_SPECIAL/OPPONENT_INK/OPPONENT_SPECIAL/
VOID/WALL`，右键会清除单格覆盖。未覆盖时仍使用 `MiniGameStage.json` 的
原始矩阵。棋盘识别器会应用这些覆盖。

当 Profile 的棋盘行列与解包数据不一致时，以 Profile 网格为准，未覆盖格默认
为可玩，由手动掩码覆盖修正。`Divide/酷似大道` 当前 Profile 使用 `18×15`，
解包数据是 `18×16`，这是已知解包数据问题。

`Painted/间隔墙` 的9个我方初始墨格和9个敌方初始墨格由关卡矩阵提供，标定器
会用不同颜色显示，运行时也会计入当前占格。

当前发现部分旧 Profile 的四点顺序不是标准的左上、右上、右下、左下，
已在 `VisionRegion` 加载时自动规范化角点顺序，避免文字 ROI 透视裁剪扭曲。

