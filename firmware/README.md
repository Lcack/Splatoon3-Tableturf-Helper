# ESP32 固件说明

## 固件文件

将 ESP32 固件文件（.bin 文件）放在此目录中。

## 获取固件

### 选项 1：使用 AutoSplatoon 的固件
1. 从 AutoSplatoon 项目获取固件文件
2. 将固件文件复制到本目录
3. 重命名为 `autotableturf.bin`

### 选项 2：编译自己的固件
1. 安装 Arduino IDE 或 PlatformIO
2. 基于 AutoSplatoon 的 ESP32 代码进行修改
3. 编译生成固件文件
4. 将固件文件放在本目录

## 烧录说明

1. **进入下载模式**：
   - 按住 ESP32 的 BOOT 按钮
   - 点击 EN 按钮
   - 释放 BOOT 按钮

2. **连接 ESP32**：
   - 通过 USB 线连接 ESP32 到电脑
   - 在网页中选择正确的串口

3. **烧录固件**：
   - 点击 "烧录固件" 按钮
   - 等待烧录完成（约 30-60 秒）

4. **重启 ESP32**：
   - 烧录完成后，点击 EN 按钮重启 ESP32
   - 或者断开 USB 重新连接

## 固件要求

固件需要实现以下功能：
- 串口通信（19200 baud）
- Switch 手柄信号模拟
- 支持 AutoSplatoon 的通信协议
- 支持 Vanilla/Chocolate 同步模式