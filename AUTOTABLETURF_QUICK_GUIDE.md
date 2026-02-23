# Autotableturf 快速使用指南

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

## 6. 实用命令总结

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

# 清理临时文件
rm -f mcts_model.pkl debug_model.pkl test_model.pkl
```

## 注意事项

1. **首次运行**需要等待JSON数据加载（约2-3秒）
2. **学习模型**自动保存在当前目录的`.pkl`文件中
3. **调试时**建议设置`learning_enabled=False`避免干扰
4. **性能问题**可减少`simulation_limit`和`time_limit_ms`

---

**保持简洁：遇到问题先看日志，再查数据，最后调参数。**