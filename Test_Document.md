# 整体体架构适合分阶段测试：
 ```
   capture → detection → model → ai → controller/manager → ui(qt/portal)
      ↓           ↓          ↓      ↓         ↓                 ↓
   视频源       图像识别   数据结构  决策算法   流程控制           界面设计
 ``` 

# 分阶段测试
## 1.capture
### capture测试方案
 ```
   capture/
   ├── __init__.py         # 不用单独运行，只是导出
   ├── interface.py        # 抽象基类，不用单独运行
   ├── file_loader.py      # 离线测试用（推荐先测这个）
   └── video_capture.py    # 需要采集卡（后测这个）
 ```
 在tests/test_capture下建立了
 ```
   test_capture/
   ├── __init__.py         
   ├── test_capture_module.py       		# 整体模块测试
   ├── test_file_loader.py      			# 测试file_loader.py
   ├── test_video_capture.py    			# 测试video_capture.py
   ├── test_video_capture_original.py       # 项目原始自带测试
   └── test_images							# 储存test_file_loader以及其他地方需要的测试图
 ```
### 测试情况
- test_file_loader.py
测试通过，测试结果：第 1 帧: shape=(1080, 1920, 3), dtype=uint8

- test_video_capture.py

## 2.detection(manager)

## 3.model

## 4.ai

## 5.controller(+manager/action)

## 6.ui(qt/portal)