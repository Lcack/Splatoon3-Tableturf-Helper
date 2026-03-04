# Autotableturf 快速使用指南

## 重要更新检查

### 1. 测试代码兼容性检查
```bash
# 检查现有测试是否兼容新代码
cd ~/Autotableturf

# 运行所有测试（检查兼容性）
python3 -m pytest tests/ -v --tb=short

# 重点检查AI相关测试
python3 -m pytest tests/test_tableturf/test_ai/ -v

# 检查MCTS AI是否正常工作
python3 -c "
from tableturf.ai.mcts import MCTSAI
import numpy as np
from tableturf.model import Status, Stage, Card, Grid

# 创建测试状态
stage = Stage(np.full((10, 10), Grid.Empty.value))
card = Card(np.zeros((8, 8), dtype=int), 3)
status = Status(stage=stage, hands=[card]*4, round=12, my_sp=0, his_sp=0)

# 测试MCTS AI
ai = MCTSAI(simulation_limit=10, learning_enabled=False)
step = ai.next_step(status)
print(f'MCTS AI测试通过: {step.action}')
"
```

**测试状态**：
- ✅ Alpha AI测试兼容（使用硬编码Card，正常）
- ⚠️ 需要添加MCTS AI专用测试
- ✅ 检测模块测试兼容（使用新card.py/stage.py）

## 1. 界面启动方式

### Qt桌面界面
```bash
# 激活conda环境
conda activate autotableturf

# 启动Qt应用
cd ~/Autotableturf
python3 -m qt.main
```

### Web界面 (Portal)
```bash
# 激活conda环境
conda activate autotableturf

# 启动Flask服务器
cd ~/Autotableturf
python3 -m portal

# 浏览器访问
# http://localhost:5000
```

## 2. 无视频源测试方法

### 使用测试图像文件
```bash
# 进入测试目录
cd ~/Autotableturf/tests

# 运行特定测试
python3 -m pytest test_tableturf/test_detection/test_card.py -v
python3 -m pytest test_tableturf/test_ai/test_alpha.py -v
python3 -m pytest test_tableturf/test_ai/test_mcts.py -v
```

### 手动测试脚本
创建测试脚本 `test_detection.py`:
```python
import cv2
import numpy as np
from tableturf.manager.detection import hands, stage_rois, stage
from tableturf.model import DataLoader, CardFactory

# 1. 加载测试图像
test_image = cv2.imread("test_screenshot.png")  # 你的截图

# 2. 测试卡牌检测
cards = hands(test_image)
print(f"检测到 {len(cards)} 张卡牌")
for i, card in enumerate(cards):
    print(f"卡牌{i}: SP消耗={card.sp_cost}, 面积={card.size}")

# 3. 测试场地检测
rois, roi_width, roi_height = stage_rois(test_image)
print(f"场地ROI: {len(rois)}个区域")

# 4. 测试JSON数据加载
loader = DataLoader()
factory = CardFactory(loader)
print(f"JSON数据: {len(loader.get_all_card_infos())}张卡牌")
```

## 3. 调试与监控方法

### 查看AI识别结果
```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 运行检测时会输出：
# [DEBUG] detection.hands: return=[Card objects...]
# [INFO] 卡牌验证完成: 3/4 匹配 (75.0%)
# [DEBUG] 卡牌0: 使用JSON匹配 (分数=0.85)
```

### 检查JSON数据使用情况
```python
from tableturf.manager.detection.card import EnhancedCardDetector
from tableturf.model import DataLoader, CardFactory

loader = DataLoader()
factory = CardFactory(loader)
detector = EnhancedCardDetector(loader, factory)

print(f"JSON支持: {detector.use_json_validation}")
print(f"加载卡牌数: {len(detector.all_cards) if detector.all_cards else 0}")

# 手动测试匹配
test_grid = np.zeros((8, 8), dtype=int)
test_grid[2:4, 2:4] = 2  # MyInk值
match, score = detector._find_best_match(test_grid, 3)
print(f"匹配测试: {match is not None}, 分数={score}")
```

### 监控MCTS学习状态
```python
from tableturf.ai.mcts import MCTSAI

# 创建带学习的AI
ai = MCTSAI(learning_enabled=True, model_path="debug_model.pkl")

# 运行一些测试后查看状态
report = ai.get_analysis_report()
print("=== MCTS学习报告 ===")
print(f"对战次数: {report['battle_stats']['total_battles']}")
print(f"胜率: {report['battle_stats']['win_rate']:.1f}%")
print(f"训练样本: {report['learning_stats']['training_samples']}")
print(f"平均误差: {report['learning_stats']['recent_avg_error']:.4f}")
print(f"决策时间: {report['performance_stats']['avg_decision_time_ms']:.0f}ms")

# 查看特征重要性
for name, importance in zip(report['feature_names'], report['feature_importance']):
    print(f"  {name}: {importance:.3f}")
```

### 强制测试降级情况
```python
# 方法1: 临时禁用JSON支持
import tableturf.manager.detection.card as card_module
card_module.HAS_JSON_SUPPORT = False

# 方法2: 创建无JSON的检测器
detector = EnhancedCardDetector(data_loader=None, card_factory=None)
print(f"降级模式: {not detector.use_json_validation}")
```

## 4. 关键配置文件

### 日志配置 (`logger/logger.py`)
```python
# 修改日志级别查看详细信息
logger.setLevel(logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR
```

### MCTS参数调整 (`tableturf/ai/mcts/mcts.py`)
```python
# 主要参数：
# - simulation_limit: 模拟次数 (默认200)
# - time_limit_ms: 思考时间 (默认1000ms)
# - learning_rate: 学习率 (默认0.01)
# - feature_dim: 特征维度 (默认10)

ai = MCTSAI(
    simulation_limit=100,      # 减少模拟加快测试
    time_limit_ms=500,        # 缩短思考时间
    learning_enabled=True,
    learning_rate=0.02,       # 提高学习率
    model_path="test_model.pkl"
)
```

## 5. 快速问题诊断

### 问题1: 检测结果不正确
```bash
# 检查依赖
python3 -c "import cv2; import numpy; print('OpenCV:', cv2.__version__)"

# 测试基础功能
python3 -c "from tableturf.model import Card; import numpy as np; grid = np.zeros((8,8)); grid[2:4,2:4]=2; c = Card(grid, 3); print('Card创建:', c.sp_cost, c.size)"
```

### 问题2: JSON数据加载失败
```bash
# 检查JSON文件
ls -la tableturf/model/*.json
python3 -c "import json; data=json.load(open('tableturf/model/MiniGameCardInfo_Matrix_ZH.json')); print('卡牌数:', len(data))"
```

### 问题3: 学习不收敛
```python
# 查看学习曲线
report = ai.get_analysis_report()
if report['learning_stats']['recent_avg_error'] > 0.5:
    print("学习误差较大，考虑：")
    print("1. 降低learning_rate到0.005")
    print("2. 增加simulation_limit到300")
    print("3. 检查特征提取是否合理")
```

## 6. UI调试与识别区域调整

### 启用调试模式查看识别区域
```python
import cv2
from tableturf.manager.detection import hands, stage_rois
from tableturf.manager.detection.debugger import OpenCVDebugger

# 1. 创建调试器
debugger = OpenCVDebugger()

# 2. 加载游戏截图
image = cv2.imread("game_screenshot.png")

# 3. 运行检测并显示调试信息
cards = hands(image, debug=debugger)  # 显示卡牌识别区域
rois, width, height = stage_rois(image, debug=debugger)  # 显示场地ROI

# 调试窗口功能：
# - 显示识别区域（绿色框）
# - 显示识别到的像素点
# - 点击像素点查看BGR/HSV值
# - 按任意键关闭窗口
```

### 识别区域常量调整（游戏UI变更时）
```python
# 文件：tableturf/manager/detection/card.py
# 主要调整参数：

# 卡牌网格位置（4张手牌）
HANDS_GRID_TOP_LEFTS = np.array([
    [192, 51],   # 手牌1左上角 (y, x)
    [203, 305],  # 手牌2
    [516, 51],   # 手牌3  
    [516, 305]   # 手牌4
])

# ROI尺寸
HANDS_GRID_ROI_WIDTH = 14    # 每个格子的宽度
HANDS_GRID_ROI_HEIGHT = 14   # 每个格子的高度

# 网格间距（自动计算用）
HANDS_GRID_ROI_WIDTH_STEPS = [25.9, 24.7, 25.9, 24.7]   # 列间距
HANDS_GRID_ROI_HEIGHT_STEPS = [23.9, 23, 23.9, 23]      # 行间距

# SP消耗显示位置
HANDS_COST_TOP_LEFTS = np.array([
    [410, 123],  # 手牌1
    [414, 373],  # 手牌2
    [731, 123],  # 手牌3
    [725, 373]   # 手牌4
])
```

### 调整步骤
1. **截取游戏画面**：在1920x1080分辨率下截图
2. **使用调试模式**：查看当前识别区域
3. **测量正确位置**：用图像工具测量实际坐标
4. **修改常量**：更新card.py中的坐标值
5. **重新测试**：验证识别准确性

### 快速测量工具
```python
import cv2
import numpy as np

def measure_coordinates(image_path):
    """快速测量图像坐标"""
    img = cv2.imread(image_path)
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"坐标: [{y}, {x}]")  # 注意：numpy是(y,x)顺序
            bgr = img[y, x]
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            print(f"颜色: BGR={bgr}, HSV={hsv}")
    
    cv2.imshow('测量工具 - 点击获取坐标', img)
    cv2.setMouseCallback('测量工具 - 点击获取坐标', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用
measure_coordinates("game_screenshot.png")
```

### 场地识别调整
```python
# 文件：tableturf/manager/detection/stage.py
# 场地边界框（需要根据游戏UI调整）
BOUNDING_BOX_TOP_LEFT = np.array([0, 750])    # 场地区域左上角
BOUNDING_BOX_WIDTH = 850                      # 场地宽度
BOUNDING_BOX_HEIGHT = 1080                    # 场地高度

# 连通组件尺寸阈值
CC_WIDTH_LOWER_BOUND = 30     # 最小宽度
CC_WIDTH_UPPER_BOUND = 45     # 最大宽度
CC_HEIGHT_LOWER_BOUND = 30    # 最小高度
CC_HEIGHT_UPPER_BOUND = 45    # 最大高度
```

## 7. 创建MCTS测试
```python
# 创建文件：tests/test_tableturf/test_ai/test_mcts.py
import unittest
import numpy as np
from tableturf.ai.mcts import MCTSAI, EnhancedMCTSAI
from tableturf.model import Status, Stage, Card, Grid

class TestMCTS(unittest.TestCase):
    def test_mcts_basic(self):
        """测试基础MCTS功能"""
        stage = Stage(np.full((10, 10), Grid.Empty.value))
        card = Card(np.zeros((8, 8), dtype=int), 3)
        status = Status(
            stage=stage,
            hands=[card]*4,
            round=12,
            my_sp=0,
            his_sp=0,
            my_deck=[],
            his_deck=[]
        )
        
        ai = MCTSAI(simulation_limit=10, learning_enabled=False)
        step = ai.next_step(status)
        self.assertIsNotNone(step)
        print(f"MCTS决策: {step.action}")
    
    def test_enhanced_mcts(self):
        """测试增强版MCTS（需要JSON数据）"""
        try:
            ai = EnhancedMCTSAI(
                npc_id="MiniGame_Aori",
                level=0,
                simulation_limit=10,
                learning_enabled=False
            )
            print("EnhancedMCTSAI创建成功")
        except Exception as e:
            print(f"EnhancedMCTSAI测试跳过: {e}")

if __name__ == '__main__':
    unittest.main()
```

## 8. 实用命令总结

```bash
# 环境检查
conda activate autotableturf
python3 --version
pip list | grep -E "numpy|opencv|flask"

# 快速测试
cd ~/Autotableturf
python3 -c "from tableturf.manager.detection import hands; print('检测模块导入成功')"
python3 -c "from tableturf.ai.mcts import MCTSAI; ai=MCTSAI(); print('MCTS AI创建成功')"

# 运行完整测试套件
python3 -m pytest tests/ -v --tb=short

# 创建MCTS测试
python3 tests/test_tableturf/test_ai/test_mcts.py

# 清理临时文件
rm -f mcts_model.pkl debug_model.pkl test_model.pkl
```

## 注意事项

1. **首次运行**需要等待JSON数据加载（约2-3秒）
2. **学习模型**自动保存在当前目录的`.pkl`文件中
3. **调试时**建议设置`learning_enabled=False`避免干扰
4. **性能问题**可减少`simulation_limit`和`time_limit_ms`
5. **UI变更时**使用调试模式查看和调整识别区域
6. **坐标顺序**：numpy使用(y,x)，OpenCV使用(x,y)

---

**调试流程**：
1. 截图 → 2. 调试模式查看 → 3. 测量坐标 → 4. 修改常量 → 5. 重新测试
**保持简洁：遇到问题先看日志，再查数据，最后调参数。**