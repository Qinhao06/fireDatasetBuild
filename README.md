# 火焰数据集生成器

这个工具可以将火焰图像随机贴到背景图像上，生成用于YOLO训练的数据集。

## 文件结构

```
fireDatasetBuild/
├── fire-photo/                    # 火焰图像目录
│   └── OIP__1_-removebg-preview.png
├── middle_photo/fire-scene-photo/  # 背景图像目录
│   ├── *.jpg                      # 背景图像文件
├── fire_dataset_generator.py      # 完整版生成器
├── simple_fire_dataset_generator.py # 简化版生成器
├── run_generator.py               # 运行脚本
└── README.md                      # 说明文档
```

## 快速开始

### 方法1：使用运行脚本（推荐）

```bash
python run_generator.py
```

运行脚本会：
1. 检查输入目录和文件
2. 询问是否安装依赖包
3. 询问生成图像数量
4. 自动选择合适的生成器版本

### 方法2：直接运行生成器

#### 安装依赖

```bash
pip install opencv-python numpy Pillow
```

#### 运行完整版生成器

```bash
python fire_dataset_generator.py
```

#### 运行简化版生成器

```bash
python simple_fire_dataset_generator.py
```

## 输出结果

生成的数据集将保存在 `fire_yolo_dataset/` 目录中：

```
fire_yolo_dataset/
├── images/                 # 生成的训练图像
│   ├── fire_dataset_000000.jpg
│   ├── fire_dataset_000001.jpg
│   └── ...
├── labels/                 # YOLO格式标注文件
│   ├── fire_dataset_000000.txt
│   ├── fire_dataset_000001.txt
│   └── ...
├── classes.txt            # 类别名称文件
└── dataset.yaml           # 数据集配置文件
```

## 功能特点

### 完整版生成器 (fire_dataset_generator.py)

- ✅ 支持PNG透明背景处理
- ✅ 随机缩放和旋转火焰图像
- ✅ 随机调整亮度和对比度
- ✅ 每张图像可添加多个火焰
- ✅ Alpha通道混合，效果更自然
- ✅ 自动生成YOLO格式标注

### 简化版生成器 (simple_fire_dataset_generator.py)

- ✅ 基础的图像合成功能
- ✅ 简单的透明背景处理
- ✅ 随机大小调整
- ✅ 每张图像添加一个火焰
- ✅ 依赖包更少，兼容性更好

## 配置参数

可以在脚本中修改以下参数：

```python
# 生成图像数量
num_images = 500

# 每张图像的火焰数量范围（仅完整版）
fires_per_image_range = (1, 2)

# 火焰大小范围（仅简化版）
target_size_range = (50, 200)

# 输出目录
output_dir = "fire_yolo_dataset"
```

## YOLO标注格式

生成的标注文件采用YOLO格式：

```
class_id x_center y_center width height
```

其中：
- `class_id`: 类别ID（火焰为0）
- `x_center, y_center`: 边界框中心点坐标（归一化）
- `width, height`: 边界框宽高（归一化）

## 用于YOLO训练

生成的数据集可以直接用于YOLOv5/YOLOv8训练：

```bash
# YOLOv5
python train.py --data fire_yolo_dataset/dataset.yaml --weights yolov5s.pt

# YOLOv8
yolo train data=fire_yolo_dataset/dataset.yaml model=yolov8n.pt
```

## 故障排除

### 常见问题

1. **ImportError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **找不到图像文件**
   - 检查 `fire-photo/` 目录是否存在
   - 检查 `middle_photo/fire-scene-photo/` 目录是否存在
   - 确保目录中有 `.jpg` 或 `.png` 文件

3. **生成的图像效果不好**
   - 尝试调整火焰图像大小范围
   - 检查火焰图像是否有透明背景
   - 使用完整版生成器获得更好效果

### 系统要求

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Pillow (可选，用于完整版)

## 自定义修改

### 添加更多火焰图像

将新的火焰图像（PNG或JPG格式）放入 `fire-photo/` 目录即可。

### 添加更多背景图像

将新的背景图像放入 `middle_photo/fire-scene-photo/` 目录即可。

### 修改生成参数

编辑生成器脚本中的参数：

```python
# 调整火焰大小范围
scale = random.uniform(0.2, 2.0)  # 缩放倍数

# 调整旋转角度范围
angle = random.uniform(-45, 45)   # 旋转角度

# 调整亮度范围
brightness = random.uniform(0.5, 1.5)  # 亮度倍数
```

## 许可证

MIT License