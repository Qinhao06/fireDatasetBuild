import cv2
import os
import glob
import numpy as np
from pathlib import Path
import shutil
import subprocess
import sys

def install_requirements():
    """安装必要的依赖"""
    try:
        import torch
        print("PyTorch 已安装")
    except ImportError:
        print("正在安装 PyTorch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    
    try:
        from ultralytics import YOLO
        print("Ultralytics 已安装")
    except ImportError:
        print("正在安装 Ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

def extract_frames_from_videos(video_dir, output_dir, frame_interval=30):
    """
    从fire-video4目录中的所有视频抽取帧
    
    Args:
        video_dir: 视频目录路径
        output_dir: 输出目录
        frame_interval: 抽帧间隔（每隔多少帧抽取一帧）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    
    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        video_files.extend(glob.glob(os.path.join(video_dir, ext.upper())))
    
    if not video_files:
        print(f"在 {video_dir} 目录下没有找到视频文件")
        return 0
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for video in video_files:
        print(f"  - {os.path.basename(video)}")
    
    total_frames_saved = 0
    
    # 处理每个视频文件
    for video_path in video_files:
        video_name = Path(video_path).stem
        print(f"\n处理视频: {video_name}")
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            continue
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
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
                output_path = os.path.join(output_dir, output_filename)
                
                # 保存帧
                cv2.imwrite(output_path, frame)
                saved_count += 1
                total_frames_saved += 1
                
                if saved_count % 20 == 0:
                    print(f"    已保存 {saved_count} 帧...")
            
            frame_count += 1
        
        cap.release()
        print(f"  完成！从 {video_name} 保存了 {saved_count} 帧")
    
    print(f"\n总共保存了 {total_frames_saved} 帧图片到 {output_dir}")
    return total_frames_saved

def detect_with_yolo_model(model_path, images_dir):
    """
    使用YOLO模型检测图片中的对象
    
    Args:
        model_path: YOLO模型文件路径
        images_dir: 图片目录
    
    Returns:
        检测结果字典
    """
    try:
        from ultralytics import YOLO
        
        # 加载YOLO模型
        print(f"加载YOLO模型: {model_path}")
        model = YOLO(model_path)
        
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        if not image_files:
            print(f"在 {images_dir} 目录下没有找到图片文件")
            return {}
        
        print(f"开始检测 {len(image_files)} 张图片...")
        
        detection_results = {}
        
        for i, image_path in enumerate(image_files):
            image_name = os.path.basename(image_path)
            
            # 进行检测
            results = model(image_path, verbose=False)
            
            # 保存检测结果
            detection_results[image_name] = results[0]
            
            if (i + 1) % 50 == 0:
                print(f"  已检测 {i + 1}/{len(image_files)} 张图片...")
        
        print(f"检测完成！共检测了 {len(image_files)} 张图片")
        return detection_results
        
    except Exception as e:
        print(f"YOLO检测过程中出错: {str(e)}")
        return {}

def convert_yolo_results_to_format(detection_results, images_dir, labels_dir, confidence_threshold=0.25):
    """
    将YOLO检测结果转换为YOLO格式的标注文件
    
    Args:
        detection_results: YOLO检测结果字典
        images_dir: 图片目录
        labels_dir: 标注文件输出目录
        confidence_threshold: 置信度阈值
    """
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"转换检测结果为YOLO格式标注文件...")
    print(f"置信度阈值: {confidence_threshold}")
    
    total_annotations = 0
    images_with_objects = 0
    
    for image_name, result in detection_results.items():
        # 获取图片尺寸
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 创建对应的标注文件名
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        # 获取检测框
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            has_valid_detection = False
            with open(label_path, 'w') as f:
                for box in boxes:
                    # 获取置信度
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # 只保存置信度大于阈值的检测结果
                    if confidence >= confidence_threshold:
                        # 获取边界框坐标 (xyxy格式)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 转换为YOLO格式 (中心点坐标和宽高，归一化)
                        center_x = (x1 + x2) / 2 / img_width
                        center_y = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        # 获取类别ID
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 写入YOLO格式: class_id center_x center_y width height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        total_annotations += 1
                        has_valid_detection = True
            
            if has_valid_detection:
                images_with_objects += 1
        else:
            # 创建空的标注文件
            with open(label_path, 'w') as f:
                pass
    
    print(f"标注转换完成！")
    print(f"  - 总图片数: {len(detection_results)}")
    print(f"  - 有目标的图片数: {images_with_objects}")
    print(f"  - 总标注数: {total_annotations}")
    
    return total_annotations

def organize_dataset(images_dir, labels_dir, dataset_dir):
    """
    组织数据集文件结构
    
    Args:
        images_dir: 原始图片目录
        labels_dir: 标注文件目录
        dataset_dir: 最终数据集目录
    """
    # 创建数据集目录结构
    train_images_dir = os.path.join(dataset_dir, "images", "train")
    train_labels_dir = os.path.join(dataset_dir, "labels", "train")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # 复制图片文件
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    for image_file in image_files:
        shutil.copy2(image_file, train_images_dir)
    
    # 复制标注文件
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    for label_file in label_files:
        shutil.copy2(label_file, train_labels_dir)
    
    print(f"数据集组织完成！")
    print(f"  - 训练图片: {len(image_files)} 张")
    print(f"  - 训练标注: {len(label_files)} 个")
    print(f"  - 数据集目录: {dataset_dir}")

def create_dataset_yaml(dataset_dir, model_path):
    """
    创建数据集配置文件
    
    Args:
        dataset_dir: 数据集目录
        model_path: 模型路径，用于获取类别信息
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        class_names = list(model.names.values())
    except:
        # 如果无法获取类别名称，使用默认值
        class_names = ['fire']
    
    yaml_content = f"""# Fire Detection Dataset Configuration
# Generated automatically from fire-video4 using {model_path}

# Dataset paths
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/train  # 使用训练集作为验证集，实际使用时应该分割数据集

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""
    
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已创建: {yaml_path}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")

def main():
    """主函数"""
    print("=" * 80)
    print("Fire Video4 YOLO数据集构建器 (使用best.pt模型)")
    print("=" * 80)
    
    # 配置路径
    video_dir = "fire-video4"
    temp_images_dir = "temp_extracted_frames"
    temp_labels_dir = "temp_labels"
    final_dataset_dir = "fire_video4_yolo_dataset"
    model_path = "best.pt"
    
    # 检查必要文件
    if not os.path.exists(video_dir):
        print(f"错误：找不到视频目录 {video_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    try:
        # 安装依赖
        print("检查并安装必要的依赖...")
        install_requirements()
        
        # 步骤1: 抽取视频帧
        print("\n步骤1: 从视频中抽取帧...")
        total_frames = extract_frames_from_videos(video_dir, temp_images_dir, frame_interval=30)
        
        if total_frames == 0:
            print("没有抽取到任何帧，程序退出")
            return
        
        # 步骤2: 使用YOLO模型检测
        print("\n步骤2: 使用YOLO模型检测对象...")
        detection_results = detect_with_yolo_model(model_path, temp_images_dir)
        
        if not detection_results:
            print("检测失败，程序退出")
            return
        
        # 步骤3: 转换为YOLO格式标注
        print("\n步骤3: 转换检测结果为YOLO格式...")
        total_annotations = convert_yolo_results_to_format(detection_results, temp_images_dir, temp_labels_dir)
        
        # 步骤4: 组织数据集结构
        print("\n步骤4: 组织数据集文件结构...")
        organize_dataset(temp_images_dir, temp_labels_dir, final_dataset_dir)
        
        # 步骤5: 创建数据集配置文件
        print("\n步骤5: 创建数据集配置文件...")
        create_dataset_yaml(final_dataset_dir, model_path)
        
        # 清理临时文件
        print("\n清理临时文件...")
        if os.path.exists(temp_images_dir):
            shutil.rmtree(temp_images_dir)
        if os.path.exists(temp_labels_dir):
            shutil.rmtree(temp_labels_dir)
        
        print("\n" + "=" * 80)
        print("数据集构建完成！")
        print(f"  - 总图片数: {total_frames}")
        print(f"  - 总标注数: {total_annotations}")
        print(f"  - 数据集目录: {final_dataset_dir}")
        print(f"  - 配置文件: {final_dataset_dir}/dataset.yaml")
        print(f"  - 使用模型: {model_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"构建过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()