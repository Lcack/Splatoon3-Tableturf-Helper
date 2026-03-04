import sys                                                                                                                        
sys.path.insert(0, 'AutoTableTurf 的绝对路径')                                                                                    
                                                                                                                                 
from capture import VideoCapture, FileLoader, Capture                                                                             
import cv2                                                                                                                        
                                                                                                                                 
def test_file_loader():                                                                                                           
   """测试离线图片加载"""                                                                                                        
   print("=" * 40)                                                                                                               
   print("测试 FileLoader")                                                                                                      
   print("=" * 40)                                                                                                               
                                                                                                                                 
   try:                                                                                                                          
       loader = FileLoader(path="./test_images/")                                                                                
       frame = loader.capture()                                                                                                  
       assert frame.shape == (1080, 1920, 3), f"尺寸错误: {frame.shape}"                                                         
       print("✅ FileLoader 正常")                                                                                               
       loader.close()                                                                                                            
       return True                                                                                                               
   except Exception as e:                                                                                                        
       print(f"❌ FileLoader 失败: {e}")                                                                                         
       return False                                                                                                              
                                                                                                                                 
def test_video_capture():                                                                                                         
   """测试采集卡"""                                                                                                              
   print("\n" + "=" * 40)                                                                                                        
   print("测试 VideoCapture")                                                                                                    
   print("=" * 40)                                                                                                               
                                                                                                                                 
   # 只测设备 0                                                                                                                  
   try:                                                                                                                          
       cap = VideoCapture(0)                                                                                                     
       frame = cap.capture()                                                                                                     
       assert frame.shape[1] == 1920 and frame.shape[0] == 1080                                                                  
       print(f"✅ VideoCapture 正常: {frame.shape}")                                                                             
       cap.close()                                                                                                               
       return True                                                                                                               
   except Exception as e:                                                                                                        
       print(f"❌ VideoCapture 失败: {e}")                                                                                       
       print("   提示：检查采集卡是否连接，或换 device_idx=1")                                                                   
       return False                                                                                                              
                                                                                                                                 
def test_interface():                                                                                                             
   """测试接口继承"""                                                                                                            
   print("\n" + "=" * 40)                                                                                                        
   print("测试 Capture 接口")                                                                                                    
   print("=" * 40)                                                                                                               
                                                                                                                                 
   # 检查 VideoCapture 是否继承 Capture                                                                                          
   assert issubclass(VideoCapture, Capture)                                                                                      
   assert issubclass(FileLoader, Capture)                                                                                        
   print("✅ 接口继承正确")                                                                                                      
   return True                                                                                                                   
                                                                                                                                 
if __name__ == "__main__":                                                                                                        
   results = []                                                                                                                  
   results.append(("FileLoader", test_file_loader()))                                                                            
   results.append(("VideoCapture", test_video_capture()))                                                                        
   results.append(("Interface", test_interface()))                                                                               
                                                                                                                                 
   print("\n" + "=" * 40)                                                                                                        
   print("测试结果汇总")                                                                                                         
   print("=" * 40)                                                                                                               
   for name, ok in results:                                                                                                      
       status = "✅ 通过" if ok else "❌ 失败"                                                                                   
       print(f"{name}: {status}") 
