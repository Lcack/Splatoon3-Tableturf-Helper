import sys                                                                                                                        
sys.path.insert(0, 'AutoTableTurf 的绝对路径')                                                                                    
                                                                                                                                     
from capture.video_capture import VideoCapture                                                                                    
                                                                                                                                     
# 测试不同设备索引                                                                                                                
for device_idx in [0, 1, 2]:                                                                                                      
    try:                                                                                                                          
        print(f"\n尝试设备索引 {device_idx}...")                                                                                  
        cap = VideoCapture(device_idx)                                                                                            
                                                                                                                                     
        print(f"  分辨率: {int(cap.width)}x{int(cap.height )}")                                                                   
                                                                                                                                     
        # 抓取一帧                                                                                                                
        frame = cap.capture()                                                                                                     
        print(f"  帧 shape: {frame.shape}")                                                                                       
                                                                                                                                     
        # 保存截图检查                                                                                                            
        cap.save(f"test_capture_device_{ device_idx}")                                                                            
        print(f"  ✅ 已保存 test_capture_device_{device_idx} .jpg")                                                               
                                                                                                                                     
        # 实时预览（按键退出）                                                                                                    
        print("  显示实时画面，按任意键退出...")                                                                                  
        cap.show("Preview")                                                                                                       
                                                                                                                                     
        cap.close()                                                                                                               
        print(f"  ✅ 设备 {device_idx} 可用！")                                                                                   
        break                                                                                                                     
                                                                                                                                     
    except Exception as e:                                                                                                        
        print(f"  ❌ 设备 {device_idx} 失败: {e}") 
