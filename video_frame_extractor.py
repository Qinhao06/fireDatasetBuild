import cv2
import os
import glob
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, frame_interval=5):
    """
    从视频中抽取帧并保存到指定目录
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_interval: 抽帧间隔（每隔多少帧抽取一帧）
    """
    # 获取视频文件名（不含扩展名）
    video_name = Path(video_path).stem
    
    # 为每个视频创建子文件夹
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"处理视频: {video_name}")
    print(f"  - 帧率: {fps:.2f} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {duration:.2f} 秒")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔抽取帧
        if frame_count % frame_interval == 0:
            # 生成输出文件名
            output_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
            output_path = os.path.join(video_output_dir, output_filename)
            
            # 保存帧
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"  已保存 {saved_count} 帧...")
        
        frame_count += 1
    
    cap.release()
    print(f"  完成！共保存 {saved_count} 帧到 {video_output_dir}")
    return saved_count

def main():
    """主函数"""
    # 当前工作目录
    current_dir = "."
    
    # 输出目录
    output_dir = "middle_photo"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.m4v']
    
    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(current_dir, ext)))
        video_files.extend(glob.glob(os.path.join(current_dir, ext.upper())))
    
    if not video_files:
        print("当前目录下没有找到视频文件")
        print(f"支持的格式: {', '.join(video_extensions)}")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for video in video_files:
        print(f"  - {os.path.basename(video)}")
    
    print(f"\n开始抽帧处理，输出目录: {output_dir}")
    print("=" * 50)
    
    total_frames_saved = 0
    
    # 处理每个视频文件
    for video_path in video_files:
        try:
            frames_saved = extract_frames_from_video(
                video_path, 
                output_dir, 
                frame_interval=30  # 每30帧抽取一帧，约每秒1帧（假设30fps）
            )
            total_frames_saved += frames_saved
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {str(e)}")
        
        print("-" * 30)
    
    print(f"\n全部完成！总共保存了 {total_frames_saved} 帧图片到 {output_dir} 目录")

if __name__ == "__main__":
    # 检查是否安装了opencv-python
    try:
        import cv2
    except ImportError:
        print("错误：需要安装 opencv-python")
        print("请运行: pip install opencv-python")
        exit(1)
    
    main()