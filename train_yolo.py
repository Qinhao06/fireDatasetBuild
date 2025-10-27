#!/usr/bin/env python3
from ultralytics import YOLO
import os

def train_yolo_model():
    # 检查数据集配置文件
    dataset_yaml = "fire_yolo_dataset/dataset.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ 数据集配置文件不存在: {dataset_yaml}")
        print("请先运行数据集生成器创建数据集")
        return
    
    print("🚀 开始训练YOLO模型...")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 或者 yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # 开始训练
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name='fire_detection',
        save=True,
        plots=True
    )
    
    print("✅ 训练完成！")
    print(f"模型保存在: runs/detect/fire_detection/weights/best.pt")

if __name__ == "__main__":
    train_yolo_model()
