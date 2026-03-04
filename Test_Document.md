# 整体架构：
 ```
   capture → detection → model → ai → controller/manager → ui(qt/portal)
      ↓           ↓          ↓      ↓         ↓                 ↓
   视频源       图像识别   数据结构  决策算法   流程控制           界面设计
 ``` 

---

# 分阶段测试：
## 1.capture模块
### capture测试方案
 ```
   capture/
   ├── __init__.py         # 不用单独运行，只是导出
   ├── interface.py        # 抽象基类，不用单独运行
   ├── file_loader.py      # 离线测试用
   └── video_capture.py    # 需要采集卡
 ```
 在tests/test_capture下建立了
 ```
   test_capture/
   ├── __init__.py
   ├── test_capture_module.py           # 整体模块测试
   ├── test_file_loader.py              # 测试file_loader.py
   ├── test_video_capture.py            # 测试video_capture.py
   └── test_images                      # 储存test_file_loader以及其他地方需要的测试图
 ```
### 测试情况
- test_file_loader.py  
  测试通过，测试输出示例：第 1 帧: shape=(1080, 1920, 3), dtype=uint8

- test_video_capture.py  
  

## 2.detection模块(manager)
### detection测试方案
 ```
   tableturf/manager/detection/
   ├── __init__.py  # 不用单独运行，只是导出
   ├── card.py      # 卡牌识别
   ├── deck.py      # 卡组识别
   ├── stage.py     # 场地识别
   ├── ui.py        # ui识别
   └── util.py      # 其他功能  
 ```

创建了以下4个测试文件：
 test_card.py - 卡牌检测测试
 - test_hands_detection_basic - 基础手牌检测
 - test_hands_detection_battle2 - 第二张对战图片测试
 - test_hands_detection_redraw - 重抽界面手牌检测
 - test_enhanced_hands_detection - 增强版检测（JSON验证）
 - test_hands_cursor_detection - 手牌光标检测
 - test_enhanced_card_detector_class - EnhancedCardDetector类测试

 test_deck.py - 卡组检测测试
 - test_deck_detection_basic - 基础卡组检测
 - test_deck_detection_with_debug - 带调试输出的检测
 - test_deck_cursor_detection - 卡组光标检测
 - test_deck_card_properties - 卡牌属性合理性检查

 test_stage.py - 场地检测测试
 - test_stage_rois_detection - ROI区域检测
 - test_stage_detection_basic - 基础场地检测
 - test_stage_with_last_stage - 使用last_stage参数
 - test_enhanced_stage_detection - 增强版场地检测
 - test_enhanced_stage_detector_class - EnhancedStageDetector类测试
 - test_sp_detection - SP点数检测
 - test_preview_detection - 卡牌预览检测

 test_ui.py - UI检测测试
 - test_deck_cursor - 卡组光标
 - test_hands_cursor - 手牌光标
 - test_redraw_cursor - 重抽光标
 - test_special_on - 特殊攻击状态
 - test_skip - 跳过按钮
 - test_replay_cursor - 回放光标
 - test_giveup_cursor - 放弃光标
 - test_result_detection - 对战结果
 - test_level_detection - 难度等级
 - test_start_detection - 开始按钮

 所有测试都使用 tests/test_capture/test_images/ 下的现有图片，并兼容原始的 temp/ 目录结构。

 额外功能：
 1. 可视化模式 (--visual 或 -v)
     - 显示检测区域（绿色方框）
     - 显示匹配比率（红色数字）
     - 鼠标点击查看 BGR/HSV 值
     - 按任意键关闭窗口
 2. ROI信息查看 (--info 或 -i)
     - 打印所有检测位置坐标
     - 显示阈值、ROI尺寸等配置

 使用方法：
 ```bash
   # 普通测试模式
   python tests/test_tableturf/test_detection/test_ui.py

   # 可视化调试模式（会弹出窗口显示检测区域）
   python tests/test_tableturf/test_detection/test_ui.py --visual
  
   # 查看ROI配置信息（打印坐标，不运行测试）
   python tests/test_tableturf/test_detection/test_ui.py --info 
 ```
 四个测试文件都支持：                             
 - test_card.py - 卡牌检测可视化
 - test_deck.py - 卡组检测可视化
 - test_stage.py - 场地检测可视化
 - test_ui.py - UI光标检测可视化（最详细，显示匹配比率）

### 测试结果
- test_card.py  
  占位符

## 3.model模块

## 4.ai模块
ai部分需要修改，enhanced alpha需要改进之后替换alpha；mcts部分代码加上集成json的代码

## 5.controller模块(包括manager/action)

## 6.ui模块(qt/portal)
