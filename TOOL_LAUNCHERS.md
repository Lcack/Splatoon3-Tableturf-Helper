# 工具启动方式

项目根目录：

```text
E:\Splatoon\tableturf\AutoTableTurf-rebuild
```

## 直接双击

```text
scripts\open_task_terminal.cmd
```

打开任务终端，用于配置NPC、目标难度、目标累计胜场、预设卡组，
并查看自动任务计划与实时进度。

```text
scripts\open_capture_debug.cmd
```

打开采集卡实时画面、录像和截图标注窗口。实时预览会叠加状态机 ROI、
棋盘格、特殊点、选牌几何和胜负区域；右侧显示 NPC、难度、地图、比分与置信度。

```text
scripts\open_video_review.cmd
```

打开视频视觉分析窗口，默认载入当前参考录像。

```text
scripts\open_calibration.cmd
```

打开高精度 ROI 标定窗口。

## 从 CMD 启动

```bat
cd /d E:\Splatoon\tableturf\AutoTableTurf-rebuild
.venv\Scripts\tableturf.exe task-terminal
```

```bat
cd /d E:\Splatoon\tableturf\AutoTableTurf-rebuild
.venv\Scripts\tableturf.exe capture-debug
```

```bat
cd /d E:\Splatoon\tableturf\AutoTableTurf-rebuild
.venv\Scripts\tableturf.exe video-review
```

```bat
cd /d E:\Splatoon\tableturf\AutoTableTurf-rebuild
.venv\Scripts\tableturf.exe calibrate-vision
```

## 从 PowerShell 启动

```powershell
Set-Location E:\Splatoon\tableturf\AutoTableTurf-rebuild
.\.venv\Scripts\tableturf.exe task-terminal
```

```powershell
Set-Location E:\Splatoon\tableturf\AutoTableTurf-rebuild
.\.venv\Scripts\tableturf.exe capture-debug
```

```powershell
Set-Location E:\Splatoon\tableturf\AutoTableTurf-rebuild
.\.venv\Scripts\tableturf.exe video-review
```

```powershell
Set-Location E:\Splatoon\tableturf\AutoTableTurf-rebuild
.\.venv\Scripts\tableturf.exe calibrate-vision
```

## 无窗口实时视觉分析

```powershell
.\.venv\Scripts\tableturf.exe vision-live `
  --backend DSHOW `
  --device-index 0 `
  --width 1920 `
  --height 1080 `
  --fps 60 `
  --stage Rectangle `
  --output artifacts\live\vision.jsonl
```

该命令不打开 Tk 或 OpenCV 窗口，只输出终端日志和 JSONL。
