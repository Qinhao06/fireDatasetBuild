# 光源负样本数据集生成器

这个工具用于生成YOLO训练的负样本数据集，将`light`文件夹中的光源图片随机贴到`middle_photo`文件夹的背景图片上。

## 功能特点

- 🔥 **智能透明处理**: 自动处理PNG图片的透明背景
- 🎯 **随机变换**: 随机缩放、旋转、位置放置光源
- 🌟 **自适应亮度**: 根据背景亮度自动调整光源亮度
- 📊 **YOLO格式**: 生成符合YOLO训练要求的数据集格式
- ⚡ **高效生成**: 快速批量生成大量负样本

## 文件说明

### 主要文件
- `light_negative_dataset_generator.py` - 完整版生成器（功能最全面）
- `simple_light_negative_generator.py` - 简化版生成器（推荐使用）
- `run_light_negative_generator.py` - 命令行运行脚本

### 输入文件夹
- `light/` - 包含光源图片（PNG格式，支持透明背景）
- `middle_photo/` - 包含背景图片（矿井场景图片）

### 输出
- `light_negative_dataset/images/` - 生成的负样本图片
- `light_negative_dataset/labels/` - 对应的空标签文件

## 使用方法

### 方法1: 直接运行简化版（推荐）

```bash
python simple_light_negative_generator.py
```

这将使用默认参数生成500个负样本。

### 方法2: 使用命令行参数

```bash
python run_light_negative_generator.py --num_samples 1000 --min_lights 1 --max_lights 3
```

参数说明：
- `--light_dir`: 光源图像文件夹路径（默认: light）
- `--background_dir`: 背景图像文件夹路径（默认: middle_photo）
- `--output_dir`: 输出文件夹路径（默认: light_negative_dataset）
- `--num_samples`: 生成的样本数量（默认: 1000）
- `--min_lights`: 每张图片最少光源数量（默认: 1）
- `--max_lights`: 每张图片最多光源数量（默认: 2）

### 方法3: 在代码中调用

```python
from simple_light_negative_generator import generate_light_negative_dataset

generate_light_negative_dataset(
    light_dir="light",
    background_dir="middle_photo", 
    output_dir="my_negative_dataset",
    num_samples=1000,
    lights_per_image=(1, 3)
)
```

## 生成效果

✅ **已成功生成**: 500个负样本
- 📁 图像文件: `light_negative_dataset/images/`
- 📁 标签文件: `light_negative_dataset/labels/`

每个负样本包含：
- 1-2个随机放置的光源
- 随机的缩放、旋转、亮度调整
- 自适应的环境光照匹配

## 数据集特点

### 负样本说明
- **目的**: 训练YOLO模型识别真正的火焰，区分光源等干扰物
- **标签**: 所有标签文件都是空的（表示图像中没有目标对象）
- **用途**: 与火焰正样本数据集合并，提高模型的准确性

### 图像处理
1. **透明背景处理**: 自动处理PNG的alpha通道
2. **随机变换**: 
   - 缩放: 5%-30% 的背景尺寸
   - 旋转: -180°到180°随机角度
   - 位置: 完全随机，避免边界截断
3. **亮度适配**: 根据背景明暗自动调整光源亮度

## 与YOLO训练集成

生成的数据集可以直接用于YOLO训练：

```python
# 合并正负样本
positive_dataset = "fire_yolo_dataset"  # 火焰正样本
negative_dataset = "light_negative_dataset"  # 光源负样本

# 合并到训练集中，建议负样本占总数据集的20-30%
```

## 技术实现

### 核心算法
- **Alpha混合**: 使用透明通道进行自然的图像混合
- **亮度分析**: 多区域采样分析背景亮度特征
- **自适应处理**: 根据环境亮度动态调整光源外观
- **冲突检测**: 避免多个光源重叠放置

### 性能优化
- 批量处理，高效I/O操作
- 内存友好的图像处理
- 异常处理，确保程序稳定运行

## 注意事项

1. **文件格式**: 光源图片建议使用PNG格式（支持透明背景）
2. **文件夹结构**: 确保`light`和`middle_photo`文件夹存在
3. **磁盘空间**: 500个样本约需要250-300MB空间
4. **处理时间**: 生成500个样本约需要1-2分钟

## 故障排除

### 常见问题
1. **找不到图像文件**: 检查文件夹路径和文件格式
2. **生成失败**: 确保有足够的磁盘空间
3. **图像质量差**: 调整亮度适配参数

### 调试模式
在代码中添加更多调试信息：
```python
print(f"处理背景: {bg_file}")
print(f"使用光源: {light_file}")
```

## 扩展功能

可以根据需要修改参数：
- 调整光源大小范围
- 修改每张图片的光源数量
- 添加更多图像增强效果
- 支持其他图像格式

---

🔥 **Happy Training!** 这个负样本数据集将帮助你的YOLO模型更好地区分真实火焰和光源干扰。