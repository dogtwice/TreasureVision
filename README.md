# TreasureVision

一个基于深度学习的计算机视觉项目，包含图像去噪、图像分类和图像相似度匹配功能。

## 项目结构

```
TreasureVision/
├── common/                 # 公共模块
│   ├── utils.py           # 工具函数
│   └── dataset/           # 数据集
├── image_denoising/       # 图像去噪模块
│   ├── denoising_config.py
│   ├── denoising_data.py
│   ├── denoising_engine.py
│   ├── denoising_model.py
│   └── denoising_train.py
├── image_classification/  # 图像分类模块
│   └── classification_test.ipynb
├── image_similarity/      # 图像相似度模块
├── image_similarity_main/ # 图像相似度主程序
├── web/                   # Web界面模块
└── .vscode/               # VS Code 配置
```

## 主要功能

### 1. 图像去噪 (Image Denoising)
- 基于深度学习的图像去噪算法
- 支持多种噪声类型的去除
- 提供训练和推理接口

### 2. 图像分类 (Image Classification)
- 支持时尚物品分类（衣服、鞋子、包包、手表等）
- 使用 PyTorch 实现
- 包含数据预处理和模型训练

### 3. 图像相似度匹配 (Image Similarity)
- 图像特征提取和相似度计算
- 支持批量图像匹配

## 环境要求

- Python 3.11+
- PyTorch
- torchvision
- PIL (Pillow)
- pandas
- numpy

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/TreasureVision.git
cd TreasureVision
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 设置环境变量：
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 使用说明

### 图像去噪
```python
from image_denoising.denoising_train import train_denoiser
train_denoiser()
```

### 图像分类
运行 `image_classification/classification_test.ipynb` 中的示例代码。

## 开发说明

- 使用 VS Code 作为开发环境
- 支持 Jupyter Notebook 开发
- 包含完整的调试配置

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

[MIT License](LICENSE)