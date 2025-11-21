import cv2
import os
from pathlib import Path

def extract_frames_from_fire_hall_video():
    """
    专门从火焰展厅环境视频中抽取帧，用于构建YOLO负样本数据集
    每3帧抽取一张图片
    """
    # 火焰展厅环境视频路径
    video_path = "火焰展厅环境.mp4"
    
    # 输出目录
    output_dir = "fire_hall_negative_frames"
    
    if not os.path.exists(video_path):
        print(f"错误：找不到视频文件 {video_path}")
        return 0
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return 0
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"处理火焰展厅环境视频:")
    print(f"  - 帧率: {fps:.2f} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 时长: {duration:.2f} 秒")
    print(f"  - 抽帧间隔: 每3帧抽取1帧")
    print(f"  - 预计抽取帧数: {total_frames // 3}")
    
    frame_count = 0
    saved_count = 0
    frame_interval = 3  # 每3帧抽取一帧
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔抽取帧
        if frame_count % frame_interval == 0:
            # 生成输出文件名
            output_filename = f"fire_hall_negative_{saved_count:06d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存帧
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"  已保存 {saved_count} 帧...")
        
        frame_count += 1
    
    cap.release()
    print(f"  完成！共保存 {saved_count} 帧到 {output_dir} 目录")
    return saved_count

def create_yolo_negative_labels(image_dir):
    """
    为负样本图片创建空的YOLO标注文件
    负样本意味着图片中没有目标对象，所以标注文件为空
    """
    labels_dir = os.path.join(image_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n为 {len(image_files)} 张负样本图片创建空标注文件...")
    
    for image_file in image_files:
        # 创建对应的标注文件名
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        
        # 创建空的标注文件（负样本没有标注内容）
        with open(label_path, 'w') as f:
            pass  # 空文件
    
    print(f"  完成！在 {labels_dir} 目录创建了 {len(image_files)} 个空标注文件")
    return len(image_files)

def main():
    """主函数"""
    print("=" * 60)
    print("火焰展厅环境视频抽帧 - YOLO负样本数据集构建")
    print("=" * 60)
    
    # 检查是否安装了opencv-python
    try:
        import cv2
    except ImportError:
        print("错误：需要安装 opencv-python")
        print("请运行: pip install opencv-python")
        return
    
    # 抽取视频帧
    saved_frames = extract_frames_from_fire_hall_video()
    
    if saved_frames > 0:
        # 创建YOLO标注文件
        create_yolo_negative_labels("fire_hall_negative_frames")
        
        print("\n" + "=" * 60)
        print("数据集构建完成！")
        print(f"  - 图片数量: {saved_frames}")
        print(f"  - 图片目录: fire_hall_negative_frames/")
        print(f"  - 标注目录: fire_hall_negative_frames/labels/")
        print("  - 数据类型: YOLO格式负样本")
        print("=" * 60)
    else:
        print("抽帧失败，请检查视频文件是否存在且可读取")

if __name__ == "__main__":
    main()