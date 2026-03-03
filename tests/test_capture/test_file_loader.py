import sys                                                                                                                        
sys.path.insert(0, 'AutoTableTurf 的绝对路径')                                                                                    
                                                                                                                                     
from capture.file_loader import FileLoader                                                                                        
                                                                                                                                     
# 测试 FileLoader                                                                                                                 
loader = FileLoader(path="./tests/test_capture/test_images/")                                                                                        
                                                                                                                                     
print(f"加载了 {len(loader._FileLoader__files)} 张图片")                                                                          
                                                                                                                                     
# 循环显示几张                                                                                                                    
import cv2                                                                                                                        
for i in range(min(3, len(loader._FileLoader__files))) :                                                                          
    frame = loader.capture()                                                                                                      
    print(f"第 {i+1} 帧: shape={frame.shape}, dtype={frame.dtype}")                                                               
                                                                                                                                     
    # 显示图片（检查是否正常）                                                                                                    
    cv2.imshow(f"Frame {i+1}", frame)                                                                                             
    cv2.waitKey(500)  # 显示500ms                                                                                                 
    cv2.destroyAllWindows()                                                                                                           
loader.close()                                                                                                                    
print("✅ file_loader.py 测试通过") 
