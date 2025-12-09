# Fire Video4 YOLO数据集

## 数据集概述
本数据集是从`fire-video4`目录中的11个火焰视频文件中提取的图片数据集，专门用于YOLO目标检测模型的训练。

## 数据集详情

### 源视频信息
- **视频数量**: 11个MP4文件
- **视频来源**: fire-video4目录
- **抽帧策略**: 每30帧抽取1帧
- **总抽取图片**: 556张

### 视频文件列表
```
Cut0_20251128095430_20251128095645.mp4
Cut0_20251128095738_20251128095900.mp4
Cut0_20251128100103_20251128100150.mp4
Cut0_20251128100245_20251128100500.mp4
Cut0_20251128100530_20251128100600.mp4
Cut0_20251128100620_20251128100652.mp4
Cut0_20251128100745_20251128100838.mp4
Cut0_20251128100843_20251128100933.mp4
Cut0_20251128101225_20251128101257.mp4
Cut0_20251128101452_20251128101510.mp4
Cut0_20251128101525_20251128101628.mp4
```

## 数据集结构
```
fire_video4_yolo_dataset/
├── dataset.yaml                    # 数据集配置文件
├── images/
│   └── train/                      # 训练图片目录
│       ├── Cut0_*_frame_000000.jpg # 抽取的视频帧图片
│       ├── Cut0_*_frame_000001.jpg
│       └── ...                     # 共556张图片
└── labels/
    └── train/                      # 训练标注目录
        ├── Cut0_*_frame_000000.txt # YOLO格式标注文件
        ├── Cut0_*_frame_000001.txt
        └── ...                     # 共556个标注文件
```

## 标注信息
- **标注格式**: YOLO格式 (class_id center_x center_y width height)
- **标注方式**: 使用best.pt模型自动检测生成
- **类别**: 根据best.pt模型的类别定义
- **置信度阈值**: 0.25

## 使用方法

### 1. 直接使用数据集
数据集已经按照YOLO标准格式组织，可以直接用于训练：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')

# 训练模型
model.train(data='fire_video4_yolo_dataset/dataset.yaml', epochs=100)
```

### 2. 重新生成标注（如果需要）
如果需要使用不同的模型或参数重新生成标注：

```bash
# 使用提供的脚本
python manual_yolo_detection.py

# 或者直接使用yolo命令
yolo predict model=best.pt source=fire_video4_yolo_dataset/images/train save_txt=True
```

## 脚本文件说明

### 主要脚本
1. **fire_video4_simple_dataset_builder.py**: 基础版本，使用OpenCV进行简单检测
2. **fire_video4_yolo_dataset_builder.py**: 完整版本，使用best.pt模型进行检测
3. **manual_yolo_detection.py**: 手动检测工具，用于重新生成标注

### 使用建议
- 如果ultralytics安装正常，使用`fire_video4_yolo_dataset_builder.py`
- 如果依赖安装有问题，可以使用`manual_yolo_detection.py`
- 对于简单测试，可以使用`fire_video4_simple_dataset_builder.py`

## 数据集特点
1. **高质量图片**: 从高清视频中提取，图片质量良好
2. **场景多样**: 涵盖不同时间段的火焰场景
3. **标准格式**: 完全符合YOLO训练格式要求
4. **自动标注**: 使用训练好的模型进行自动标注，减少人工工作量

## 注意事项
1. 标注文件是基于best.pt模型的检测结果生成的，准确性取决于该模型的性能
2. 建议在实际使用前对部分标注进行人工验证和修正
3. 可以根据需要调整置信度阈值来过滤检测结果
4. 如需更高质量的标注，建议使用人工标注工具如LabelImg进行精细标注

## 扩展使用
- 可以与其他火焰检测数据集合并使用
- 可以作为预训练数据集的一部分
- 可以用于数据增强的基础数据

## 技术规格
- **图片格式**: JPG
- **图片尺寸**: 原视频尺寸（各视频可能不同）
- **标注格式**: YOLO v5/v8兼容
- **编码**: UTF-8
- **操作系统**: 跨平台兼容

---
*数据集生成时间: 2024年*
*生成工具: 自定义Python脚本*
*模型版本: best.pt*