import cv2
import os
import glob
import numpy as np
from pathlib import Path
import shutil

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

def detect_fire_regions_opencv(image_path):
    """
    使用OpenCV进行简单的火焰区域检测
    基于颜色和亮度特征来识别可能的火焰区域
    
    Args:
        image_path: 图片路径
    
    Returns:
        检测到的边界框列表 [(x1, y1, x2, y2, confidence), ...]
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    height, width = img.shape[:2]
    
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义火焰的HSV范围
    # 火焰通常是红色、橙色、黄色
    lower_fire1 = np.array([0, 50, 50])    # 红色范围1
    upper_fire1 = np.array([10, 255, 255])
    
    lower_fire2 = np.array([170, 50, 50])  # 红色范围2
    upper_fire2 = np.array([180, 255, 255])
    
    lower_fire3 = np.array([10, 50, 50])   # 橙色/黄色范围
    upper_fire3 = np.array([30, 255, 255])
    
    # 创建掩码
    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    mask3 = cv2.inRange(hsv, lower_fire3, upper_fire3)
    
    # 合并掩码
    fire_mask = cv2.bitwise_or(mask1, mask2)
    fire_mask = cv2.bitwise_or(fire_mask, mask3)
    
    # 添加亮度条件
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    
    # 结合颜色和亮度掩码
    fire_mask = cv2.bitwise_and(fire_mask, bright_mask)
    
    # 形态学操作去除噪声
    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 过滤太小的区域
        if area < 500:  # 最小面积阈值
            continue
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算置信度（基于面积和形状）
        confidence = min(0.9, area / (width * height) * 10)  # 简单的置信度计算
        
        if confidence > 0.3:  # 置信度阈值
            detections.append((x, y, x + w, y + h, confidence))
    
    return detections

def process_images_with_opencv_detection(images_dir):
    """
    使用OpenCV对所有图片进行火焰检测
    
    Args:
        images_dir: 图片目录
    
    Returns:
        检测结果字典
    """
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
        detections = detect_fire_regions_opencv(image_path)
        detection_results[image_name] = detections
        
        if (i + 1) % 50 == 0:
            print(f"  已检测 {i + 1}/{len(image_files)} 张图片...")
    
    print(f"检测完成！共检测了 {len(image_files)} 张图片")
    return detection_results

def convert_to_yolo_format(detection_results, images_dir, labels_dir):
    """
    将检测结果转换为YOLO格式的标注文件
    
    Args:
        detection_results: 检测结果字典
        images_dir: 图片目录
        labels_dir: 标注文件输出目录
    """
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"转换检测结果为YOLO格式标注文件...")
    
    total_annotations = 0
    images_with_objects = 0
    
    for image_name, detections in detection_results.items():
        # 获取图片尺寸
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 创建对应的标注文件名
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        if detections:
            images_with_objects += 1
            with open(label_path, 'w') as f:
                for detection in detections:
                    x1, y1, x2, y2, confidence = detection
                    
                    # 转换为YOLO格式 (中心点坐标和宽高，归一化)
                    center_x = (x1 + x2) / 2 / img_width
                    center_y = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # 类别ID (0表示火焰)
                    class_id = 0
                    
                    # 写入YOLO格式: class_id center_x center_y width height
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    total_annotations += 1
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

def create_dataset_yaml(dataset_dir, class_names=None):
    """
    创建数据集配置文件
    
    Args:
        dataset_dir: 数据集目录
        class_names: 类别名称列表
    """
    if class_names is None:
        # 默认火焰检测类别
        class_names = ['fire']
    
    yaml_content = f"""# Fire Detection Dataset Configuration
# Generated automatically from fire-video4

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

def main():
    """主函数"""
    print("=" * 80)
    print("Fire Video4 YOLO数据集构建器 (OpenCV版本)")
    print("=" * 80)
    
    # 配置路径
    video_dir = "fire-video4"
    temp_images_dir = "temp_extracted_frames"
    temp_labels_dir = "temp_labels"
    final_dataset_dir = "fire_video4_yolo_dataset"
    
    # 检查必要文件
    if not os.path.exists(video_dir):
        print(f"错误：找不到视频目录 {video_dir}")
        return
    
    try:
        # 步骤1: 抽取视频帧
        print("\n步骤1: 从视频中抽取帧...")
        total_frames = extract_frames_from_videos(video_dir, temp_images_dir, frame_interval=30)
        
        if total_frames == 0:
            print("没有抽取到任何帧，程序退出")
            return
        
        # 步骤2: 使用OpenCV检测火焰区域
        print("\n步骤2: 使用OpenCV检测火焰区域...")
        detection_results = process_images_with_opencv_detection(temp_images_dir)
        
        if not detection_results:
            print("检测失败，程序退出")
            return
        
        # 步骤3: 转换为YOLO格式标注
        print("\n步骤3: 转换检测结果为YOLO格式...")
        total_annotations = convert_to_yolo_format(detection_results, temp_images_dir, temp_labels_dir)
        
        # 步骤4: 组织数据集结构
        print("\n步骤4: 组织数据集文件结构...")
        organize_dataset(temp_images_dir, temp_labels_dir, final_dataset_dir)
        
        # 步骤5: 创建数据集配置文件
        print("\n步骤5: 创建数据集配置文件...")
        create_dataset_yaml(final_dataset_dir)
        
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
        print("\n注意：此版本使用OpenCV进行简单的火焰检测")
        print("如需使用训练好的YOLO模型，请安装ultralytics和torch")
        print("=" * 80)
        
    except Exception as e:
        print(f"构建过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()