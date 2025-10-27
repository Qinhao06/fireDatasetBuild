import cv2
import os
import glob

def extract_frames():
    """简单的视频抽帧脚本"""
    
    # 创建输出文件夹
    output_dir = "middle_photo"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 查找当前目录下的视频文件
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(ext))
    
    if not video_files:
        print("没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    for video_file in video_files:
        print(f"正在处理: {video_file}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_file)
        
        # 获取视频名称（去掉扩展名）
        video_name = os.path.splitext(video_file)[0]
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每30帧保存一张图片
            if frame_count % 30 == 0:
                filename = f"{video_name}_frame_{saved_count:04d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"  保存了 {saved_count} 张图片")
    
    print("抽帧完成！")

if __name__ == "__main__":
    extract_frames()